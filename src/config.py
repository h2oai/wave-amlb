from h2o_wave import ui
from collections import defaultdict
import os

uploaded_files_dict = defaultdict()
# Card placements across the app
# Order of elements -> offset left, offset top, width, height

logo_file = 'wave_logo.png'
cur_dir = os.getcwd()

# to uploaded demo files in app. can add results_test.csv here if useful
uploaded_files_dict['results_demo.csv'] = [f'{cur_dir}/data/results_demo.csv']

class Configuration:
    """
    Configuration file Data Labeling
    """
    def __init__(self, env='prod'):
        self._env = env
        self.title = 'AutoML Benchmark'
        self.subtitle = 'Wave UI for OpenML AutoML Benchmark'
        self.icon = 'Settings'
        self.icon_color = '$yellow'
        self.default_title = ui.text_xl('H2O-3 UI')
        self.items_guide_tab = [
            ui.text("""<center><img width="650" height="300" src="https://i0.wp.com/blog.okfn.org/files/2017/12/openml-logo.png?fit=1175%2C537&ssl=1"></center>"""),
            ui.text("""
            
This Wave application allows users to visualize their OpenML AutoML Benchmark runs via the Wave UI. 

References: 
- https://openml.github.io/automlbenchmark/
- https://github.com/openml/automlbenchmark
            """),
        ]
        self.banner_box = '1 1 3 1'
        self.logo_box = '3 1 1 1'
        self.navbar_box = '4 1 -1 1'
        self.main_box = '1 2 -1 -1'
        self.plot1_box = '1 2 -1 3'
        self.plot21_box = '1 5 -1 7'
        self.plot22_box = '1 12 -1 11'
        self.tmp_dir = '/tmp'
