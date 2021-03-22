from h2o_wave import Q, ui, app, main, data
from .config import *
import time
from datetime import date
import pandas as pd
import numpy as np
import base64
import io
from functools import reduce
import os
import sys
import re

from report import draw_score_heatmap, draw_score_parallel_coord, draw_score_pointplot, draw_score_stripplot, draw_score_barplot, prepare_results, render_leaderboard, render_metadata, render_summary
from report.config import *
from report.util import create_file, display
from report.visualizations.util import register_colormap, render_colormap, savefig
from report.metadata import load_dataset_metadata

app_config = Configuration()

@app('/')
async def serve(q: Q):

    # creating a temp directory for the app
    cur_dir = os.getcwd()
    q.app.tmp_dir = cur_dir + app_config.tmp_dir
    if not os.path.exists(q.app.tmp_dir):
        os.mkdir(q.app.tmp_dir)

    # Hash routes user when tabs are clicked
    hash = q.args['#']
    if hash == 'home':
        await clean_cards(q)
        await main_menu(q)
    elif hash == 'import':
        await clean_cards(q)
        await import_menu(q)
    elif hash == 'parameters':
        await clean_cards(q)
        await select_table(q)
    elif hash == 'benchmark_report':
        await clean_cards(q)
        await show_plots(q)
    # User uploaded a results csv 
    elif q.args.uploaded_file:
        await upload_data(q)
    # User selected results csv
    elif q.args.selected_parameters_next:
        await parameters_selection_menu(q)
    # User selected the parameters
    elif q.args.next_generate_report:
        await start_report(q)
    else:
        await main_menu(q)
    await q.page.save()

# main menu for home guide
async def main_menu(q: Q):
    q.app.df_rows = []
    q.page['banner'] = ui.header_card(box=app_config.banner_box, title=app_config.title, subtitle=app_config.subtitle,
                                      icon=app_config.icon, icon_color=app_config.icon_color)

    q.page['menu'] = ui.toolbar_card(
        box=app_config.navbar_box,
        items=[
            ui.command(name="#home", label="Home", caption="Home", icon="Home"),
            ui.command(name="#import", label="Import Data", caption="Import Data", icon="Database"),
            ui.command(name="#parameters", label="Parameters Selection", caption="Parameters Selection", icon="BullseyeTarget"),
            ui.command(name="#benchmark_report", label="Benchmark Comparison Report", caption="Benchmark Report", icon="ClipboardList"),
        ])

    # Logo
    if not q.app.logo_url:
        q.app.logo_url, = await q.site.upload([logo_file])
    q.page['main'] = ui.form_card(box=app_config.main_box, 
    items=[
        # app_config.items_guide_tab,
        ui.text("""<center><img width="650" height="300" src="https://i0.wp.com/blog.okfn.org/files/2017/12/openml-logo.png?fit=1175%2C537&ssl=1"></center>"""),
        ui.text("""
This Wave application allows users to visualize their OpenML AutoML Benchmark runs via the Wave UI. 

References: 
- https://openml.github.io/automlbenchmark/
- https://github.com/openml/automlbenchmark
            """),
        ui.text_l("""
**How to use this app**
1. Click on the import data tab
2. On the import data tab, upload a benchmark results csv 
3. Select the results csv you uploaded
4. On the parameters selection you can select the parameters for which you would like to evaluate your results 
5. After selecting your parameters, the app will begin to generate a table and visualizations for you to analyze
            """)
    ])

    await q.page.save()

# menu for importing new results csv on import data
async def import_menu(q: Q):
    q.page['main'] = ui.form_card(box=app_config.main_box, items=[
        ui.text_xl('Import Data'),
        ui.file_upload(name='uploaded_file', label='Click to upload selected files!', file_extensions=['csv'],
        multiple=False),
    ])

# func for uploading the results csv
async def upload_data(q: Q):
    # q.args.uploaded_file comes from uploading in import menu
    uploaded_file_path = q.args.uploaded_file
    for file in uploaded_file_path:
        filename = file.split('/')[-1]
        uploaded_files_dict[filename] = uploaded_file_path
    time.sleep(1)
    q.page['main'] = ui.form_card(box=app_config.main_box,
                                  items=[ui.message_bar('success', 'File Imported! Please select an action'),
                                         ui.buttons([ui.button(name='#parameters', label='Select parameters', primary=True),
                                                    # main menu takes you back to the #guide which is home
                                                     ui.button(name='#home', label='Main Menu', primary=False)])])

# from dropdown confirm the results csv to use
async def select_table(q: Q, warning: str = ''):
    choices = []
    if uploaded_files_dict:
        for file in uploaded_files_dict:
            choices.append(ui.choice(file, file))
        q.page['main'] = ui.form_card(box=app_config.main_box, items=[
            ui.message_bar(type='warning', text=warning),
            ui.dropdown(name='results_file', label='Select Results CSV For Parameter Selection', value=q.app.results_file, required=True,
                        choices=choices),
            ui.buttons([ui.button(name='selected_parameters_next', label='Next', primary=True)])
        ])

    else:
        # if someone goes to the parameters selection without importing data
        q.page['main'] = ui.form_card(box=app_config.main_box, items=[
            ui.text_xl(f'{q.app.task}'),
            ui.message_bar(type='warning', text=warning),
            ui.text(f'No data found. Please import data first.'),
            ui.buttons([ui.button(name='#import', label='Import Data', primary=True)])
        ])

# selecting the parameters to parse the results csv by
async def parameters_selection_menu(q: Q, warning: str = ''):
    # Error handling
    if not q.args.results_file and not q.app.results_file:
        await select_table(q, 'Please Select A Results CSV To Procceed')
        return

    # Store results file
    if q.args.results_file:
        q.app.results_file = q.args.results_file
        # Check for data provided as part of app
        if 'data/results_demo.csv' in uploaded_files_dict[q.app.results_file][0]:
            local_path = uploaded_files_dict[q.app.results_file][0]
        else:
            # uploading the file that we decieded to import 
            local_path = await q.site.download(uploaded_files_dict[q.app.results_file][0], '.')
        # creates a pandas df out of that file 
        q.app.results_df = pd.read_csv(local_path)
    
    # if they didn't upload a results csv file send them to import data
    if not {'framework', 'constraint', 'mode', 'task'}.issubset(q.app.results_df.columns):
        q.page['meta'] = ui.meta_card(box ='')
        q.page['main'] = ui.markdown_card(box=app_config.main_box, title ='Incorrect Results CSV',
                                          content='This file does not match the results.csv format. Please upload another file'
        )
        await q.page.save()
        time.sleep(10)
        q.page['meta'].redirect = '#import'
        # await import_menu(q, 'This file does not match the results.csv format. Please upload another file')
        return

    # choices based on the results csv uploaded
    framework_choices = [ui.choice(i, i) for i in list(q.app.results_df['framework'].unique())]
    constraint_choices = [ui.choice(i, i) for i in list(q.app.results_df['constraint'].unique())]
    mode_choices = [ui.choice(i, i) for i in list(q.app.results_df['mode'].unique())]
    problem_choices = [ui.choice(i, i) for i in ['binary', 'multiclass', 'regression']]
    
    q.page['main'] = ui.form_card(box=app_config.main_box, items=[
        ui.text_xl(f'Benchmark Comparison Parameters'),
        ui.message_bar(type='warning', text=warning),
        ui.dropdown(name='frameworks', label='Frameworks', placeholder='Example: H2OAutoML, lightautoml',
                    values=[], required=True, choices=framework_choices),
        # ui.dropdown(name='ref_framework', label='Reference Framework', placeholder='Example: H2OAutoML',
        #             value=q.app.ref_framework, required=True, choices=framework_choices),
        ui.dropdown(name='constraint', label='Constraint', placeholder='Example: 1h8c',
                    value=q.app.constraint, required=True, choices=constraint_choices),
        ui.dropdown(name='mode', label='Mode', placeholder='Example: aws',
                    value=q.app.mode, required=True, choices=mode_choices),
        ui.dropdown(name='problem_type', label='Problem Type', placeholder='Example: binary',
                    value=q.app.problem_type, required=True, choices=problem_choices),
        ui.separator(label='OpenML Datasets Filters'),
        ui.textbox(name='max_rows_lower_bound',
                   label='Rows Lower Bound', value='1'),
        ui.textbox(name='max_rows_upper_bound',
                   label='Rows Upper Bound', value='inf'),
        ui.textbox(name='max_features_lower_bound',
                   label='Features Lower Bound', value='1'),
        ui.textbox(name='max_features_upper_bound',
                   label='Features Upper Bound', value='inf'),
        ui.textbox(name='max_cardinality_lower_bound', label='Max Cardinality Lower Bound', value='0'),
        ui.textbox(name='max_cardinality_upper_bound', label='Max Cardinality Upper Bound', value='inf'),
        ui.buttons([ui.button(name='next_generate_report', label='Next', primary=True)])
    ])

# this where we will start the benchmark report and check that at least two frameworks have been selected
async def start_report(q: Q):

    q.page['main'] = ui.form_card(box=app_config.main_box, items=[])
    # if at least two frameworks haven't been selected send them back to choose frameworks in parameters
    if len(q.args.frameworks) < 2:
        await parameters_selection_menu(q, 'Please Select At Least Two Frameworks For Benchmark Comparison Report')
        return

    # Tell user that your initating the data visualizations of the benchmark report
    main_page = q.page['main']
    main_page.items = [ui.progress(label='Generating Data Visualizations for Benchmark Report')]
    await q.page.save()

    # we would be calling to show the matplotlib plots
    await show_plots(q)

# func to generate a table from pandas df
def table_from_df(df: pd.DataFrame, table_name: str):
    # Columns for the table
    columns = [ui.table_column(
        name=str(x.replace('_',' ')),
        label=str(x.replace('_', ' ')),
        sortable=True,  # Make column sortable
        filterable=True,  # Make column filterable
        searchable=False,  # Make column searchable
        data_type= (np.where(re.search(r'mean|std|folds|rows|features|cardinality|imbalance', x), 'number', 'string')).item()
    ) for x in df.columns.values]
    # Rows for the table
    rows = [ui.table_row(name=str(i), cells=[str(cell) for cell in row]) for i, row in df.iterrows()] 
    table = ui.table(name=f'{table_name}',
             rows=rows,
             columns=columns,
             multiple=True, # Hack for not navigating alway when doing 1 click on table
             downloadable  = True, 
             height='500px')
    return table

# create the metadata filter for openml datasets
def metadata_filter(results_df, max_cardinality_lower_bound, max_cardinality_upper_bound, max_rows_lower_bound, max_rows_upper_bound, max_features_lower_bound, max_features_upper_bound):
    metadata = load_dataset_metadata(results_df)
    metadata_df = render_metadata(metadata)
    # max cardinality filter
    tmp_md = metadata_df[(metadata_df['max_cardinality'] > max_cardinality_lower_bound) & (metadata_df['max_cardinality'] < max_cardinality_upper_bound)]
    # max rows filter
    tmp_md = tmp_md[(tmp_md['nrows'] > max_rows_lower_bound) & (tmp_md['nrows'] < max_rows_upper_bound)]
    # max features filter 
    tmp_md = tmp_md[(tmp_md['nfeatures'] > max_features_lower_bound) & (tmp_md['nfeatures'] < max_features_upper_bound)]
    metadata_task_filter = tmp_md['task'].tolist()
    return metadata_task_filter

# create the benchmark df and the results csv 
def create_benchmark_df(results_df, framework, constraint, mode, problem_type, metadata_filter):
    # filter the results csv by framework, constraint, mode
    benchmark_df = results_df[(results_df['framework'] == framework) & (
        results_df['constraint'] == constraint) & (results_df['mode'] == mode)]
    # filter by problem type
    if problem_type == 'binary':
        benchmark_df = benchmark_df[benchmark_df['metric'] == 'auc']
    elif problem_type == 'multiclass':
        benchmark_df = benchmark_df[benchmark_df['metric'] == 'logloss']
    else:
        benchmark_df = benchmark_df[benchmark_df['metric'] == 'rmse']
    # filter by metadata
    benchmark_df = benchmark_df[benchmark_df['id'].isin(metadata_filter)]
    # save the benchmark_df
    benchmark_df_csv_path = f"{cur_dir}/tmp/results_{framework}_{date.today()}.csv"
    benchmark_df.to_csv(benchmark_df_csv_path)
    return benchmark_df_csv_path

# results as df to remove any row filter 
def results_as_df(results_dict, row_filter=None):
    def apply_filter(res, filtr):
        r = res.results
        return r.loc[filtr(r)]
    if row_filter is None:
        row_filter = lambda r: True
    return pd.concat([apply_filter(res, lambda r: (r.framework == name) & row_filter(r))
                      for name, res in results_dict.items()
                      if res is not None])

# definitions dictionary for matplotlib plots creation
def create_definitions_dict(frameworks, results_df, constraint, mode, problem_type, metadata_filter):
    definitions = dict()
    # assign the reference framework to the first framework in list
    ref_framework = frameworks[0]
    for framework in frameworks:
        # generate the results.csv path
        benchmark_csv_path = create_benchmark_df(
            results_df, framework, constraint, mode, problem_type, metadata_filter)
        # check if the current framework is the reference framework
        if framework == ref_framework:
            # include ref = true in dict
            definitions[str.lower(framework)] = dict(
                ref=True, framework=framework, results_files=[benchmark_csv_path])
        else:
            # add to the definitions dictionary this framework
            definitions[str.lower(framework)] = dict(
                framework=framework, results_files=[benchmark_csv_path])
    return definitions                     

# creating the dataframe that contains info for benchmark report table
def benchmark_report_table(col, results, metadata):
    # score data
    df = results.dropna(subset=['id']).groupby(['task', 'framework']).agg(
        mean_score =(col, "mean"),
        std_deviation =(col, "std"),
        folds = ('fold',"size")
    ).reset_index()
    df['mean_score'] = df['mean_score'].round(5)
    df['std_deviation'] = df['std_deviation'].round(5)
    # metadata df
    metadata_df = render_metadata(metadata)
    metadata_df = metadata_df[['name', 'nrows', 'nfeatures', 'max_cardinality','class_imbalance']]
    metadata_df['class_imbalance'] = metadata_df['class_imbalance'].round(5)
    metadata_df.rename(columns={'nrows':'rows','nfeatures':'features'}, inplace = True)
    metadata_df['name'] = metadata_df['name'].str.lower()
    # merge the df rows, features, max cardinality
    df = pd.merge(df, metadata_df, left_on= 'task',right_on='name', how = 'left')
    df.drop(columns='name',inplace = True)
    df = df.astype({'features': 'int64','max_cardinality':'int64'})
    return df



# Show the plots! 
async def show_plots(q: Q):

    # check to make sure there are frameworks selected
    if q.args.frameworks:

        # items to be defined in a function or config file 
        excluded_frameworks = []
        duplicates_handling = 'fail'
        frameworks_labels = None
        imputation = None
        normalization = None
        row_filter = None
        title_extra = ""
        output_dir = "./tmp"

        
        # create the metadata filter 
        metadata_task_filter = metadata_filter(results_df=q.app.results_df, max_cardinality_lower_bound=int(q.args.max_cardinality_lower_bound), 
        max_cardinality_upper_bound=float(q.args.max_cardinality_upper_bound), max_rows_lower_bound=int(q.args.max_rows_lower_bound), 
        max_rows_upper_bound=float(q.args.max_rows_upper_bound), max_features_lower_bound=int(q.args.max_features_lower_bound), max_features_upper_bound=float(q.args.max_features_upper_bound))

        # create the definitions dict 
        definitions = create_definitions_dict(frameworks = q.args.frameworks, results_df = q.app.results_df, 
                                              constraint=q.args.constraint, mode=q.args.mode, 
                                              problem_type=q.args.problem_type, metadata_filter=metadata_task_filter)

        # runs is equivalent to definitions dict because we arent excluding anything 
        runs = definitions

        # ref results
        ref_results = {name: prepare_results(run['results_files'],
                                    renamings={run['framework']: name},
                                    exclusions=excluded_frameworks,
                                    normalization=normalization,
                                    duplicates_handling=duplicates_handling,
                                    )
            for name, run in runs.items() if runs[name].get('ref', False)}

        # ref results and row filter combined if there are any filters
        all_ref_res = results_as_df(ref_results, row_filter)

        # create the run results 
        runs_results = {name: prepare_results(run['results_files'],
                                                renamings={run['framework']: name},
                                                exclusions=excluded_frameworks,
                                                imputation=imputation,
                                                normalization=normalization,
                                                ref_results=all_ref_res,
                                                duplicates_handling=duplicates_handling
                                                )
                        for name, run in runs.items() if name not in ref_results}

        # create the final all res for plotting, this has all the results
        all_res = pd.concat([
            all_ref_res,
            results_as_df(runs_results, row_filter)
        ])

        # this is a dict that contains the metadata for all the datasets
        metadata = reduce(lambda l, r: {**r, **l},
                        [res.metadata
                        for res in list(ref_results.values())+list(runs_results.values())
                        if res is not None],
                        {})

        # add the problem types
        problem_types = pd.DataFrame(m.__dict__ for m in metadata.values())['type'].unique().tolist()
        
        # creates a df of score summary to be used in the sorting function for plots
        score_summary = render_summary('score', results=all_res)

        # grab the reference framework
        def reference_framework(definitions_dict=definitions):
            return next(iter(definitions))
        
        # create the sorting function for plots
        def tasks_sort_by_score(df):
            ref_framework_name = reference_framework()
            return [score_summary.loc[score_summary.index.get_level_values('task') == row['task']].iloc[0].at[ref_framework_name] for _, row in df.iterrows()]
    
        if 'binary' in problem_types:
            fig_stripplot = draw_score_stripplot('score',
                                    results=all_res.sort_values(by=['framework']),
                                    type_filter='binary',
                                    metadata=metadata,
                                    xlabel=binary_score_label,
                                    y_sort_by=tasks_sort_by_score,
                                    hue_sort_by=frameworks_sort_key,
                                    title=f"Scores ({binary_score_label}) on {results_group} binary classification problems{title_extra}",
                                    legend_labels=frameworks_labels,
                                    )

            fig_pointplot = draw_score_pointplot('score',
                                    results=all_res,
                                    type_filter='binary', 
                                    metadata=metadata,
                                    x_sort_by=tasks_sort_by_score,
                                    ylabel=binary_score_label,
                                    ylim=dict(bottom=.5),
                                    hue_sort_by=frameworks_sort_key,
                                    join='none', marker='hline_xspaced', ci=95, 
                                    title=f"Scores ({binary_score_label}) on {results_group} binary classification problems{title_extra}",
                                    legend_loc='lower center',
                                    legend_labels=frameworks_labels,
                                    )
        if 'multiclass' in problem_types:
            fig_stripplot = draw_score_stripplot('score',
                                    results=all_res.sort_values(by=['framework']),
                                    type_filter='multiclass',
                                    metadata=metadata,
                                    xlabel=multiclass_score_label,
                                    xscale='symlog',
                                    y_sort_by=tasks_sort_by_score,
                                    hue_sort_by=frameworks_sort_key,
                                    title=f"Scores ({multiclass_score_label}) on {results_group} multi-class classification problems{title_extra}",
                                    legend_labels=frameworks_labels,
                                    )
            fig_pointplot = draw_score_pointplot('score',
                                    results=all_res,
                                    type_filter='multiclass',
                                    metadata=metadata,
                                    x_sort_by=tasks_sort_by_score,
                                    ylabel=multiclass_score_label,
                                    hue_sort_by=frameworks_sort_key,
                                    join='none', marker='hline_xspaced', ci=95,
                                    title=f"Scores ({multiclass_score_label}) on {results_group} multiclass classification problems{title_extra}",
                                    legend_loc='lower center',
                                    legend_labels=frameworks_labels,
                                    )
        if 'regression' in problem_types:
            fig_stripplot = draw_score_stripplot('score',
                                    results=all_res,
                                    type_filter='regression',
                                    metadata=metadata,
                                    xlabel=regression_score_label,
                                    xscale='symlog',
                                    y_sort_by=tasks_sort_by_score,
                                    hue_sort_by=frameworks_sort_key,
                                    title=f"Scores ({regression_score_label}) on {results_group} regression problems{title_extra}",
                                    legend_labels=frameworks_labels,
                                    )
            fig_pointplot = draw_score_pointplot('score',
                                    results=all_res,
                                    type_filter='regression', 
                                    metadata=metadata,
                                    x_sort_by=tasks_sort_by_score,
                                    ylabel=regression_score_label,
                                    yscale='symlog',
                                    ylim=dict(top=0.1),
                                    hue_sort_by=frameworks_sort_key,
                                    join='none', marker='hline_xspaced', ci=95, 
                                    title=f"Scores ({regression_score_label}) on {results_group} regression classification problems{title_extra}",
                                    legend_loc='lower center',
                                    legend_labels=frameworks_labels,
                                    size=(8, 6),
                                    )       

        # create data for the benchmark table
        benchmark_metadata_df = benchmark_report_table('score', all_res, metadata)

        # create benchmark table
        benchmark_metadata_table = table_from_df(
            benchmark_metadata_df, 'benchmark_metadata_table')

        # add the plots to the page 
        q.page['main'] = ui.form_card(box=app_config.plot1_box, items=[
            ui.text_xl(f'Benchmark Comparison Report: {q.args.problem_type}'),
            benchmark_metadata_table])


        q.page['plot22'] = ui.image_card(
            box=app_config.plot22_box,
            title="Strip Plot",
            type="png",
            image=get_image_from_matplotlib(fig_stripplot),
            )
        
        q.page['plot21'] = ui.image_card(
            box=app_config.plot21_box,
            title="Point Plot",
            type="png",
            image=get_image_from_matplotlib(fig_pointplot),
            )

    else:
        # When a user goes straight to the benchmark comparison report without
        # selecting parameters to create report
        q.page['main'] = ui.form_card(box=app_config.main_box, items=[
            ui.text_xl('Benchmark Comparison Report'),
            ui.text('No benchmarks to compare. Please select benchmark comparison parameters first.'),
            ui.buttons([ui.button(name='#parameters', label='Select parameters', primary=True)])
        ])

# Clean cards
async def clean_cards(q: Q):
    cards_to_clean = ['plot1', 'plot21', 'plot22']
    for card in cards_to_clean:
        del q.page[card]

# Image from matplotlib object
def get_image_from_matplotlib(matplotlib_obj):
    buffer = io.BytesIO()
    # buffer is an in-memory object that can be used wherever a file is used. 
    matplotlib_obj.tight_layout()
    matplotlib_obj.savefig(buffer, format="png", bbox_inches='tight')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

