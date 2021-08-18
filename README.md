# AutoML Benchmark Wave App

Dashboard to compare AutoML frameworks by exploring the results from runs of the [OpenML AutoML Benchmark](https://github.com/openml/automlbenchmark).

**App Goal:** View & compare performance of different AutoML frameworks, as benchmarked on the OpenML AutoML Benchmark

**Target Audience:** Data scientists

**Industry:** Any

**GitHub:** [https://github.com/h2oai/wave-amlb](https://github.com/h2oai/wave-abml)

**Actively Being Maintained:** Yes

**Comes with Demo Mode (pre-loaded data, models, results, etc.):** Yes

**Allows for Uploading and Using New Data:** Yes

**Detailed Description:** 

TO DO

## System Requirements 
1. Python 3.6+
2. pip3

## Installation 

### 1. Run the Wave Server
Follow the instructions [here](https://h2oai.github.io/wave/docs/installation) to download and run the latest Wave Server, a requirement for apps. 

### Setting up Wave for the first time

1. Go to [download](https://github.com/h2oai/wave/releases/tag/v0.12.0) the wave SDK for your platform
    - Make sure to scroll down on the page to Assets. For MacOs select "darwin".
```
wave-0.12.0-darwin-amd64.tar.gz
```

2. Extract your download:
    - Make sure to run this command in the directory you downloaded the <file>.tar.gz
```
tar -xzf wave-0.12.0-darwin-amd64.tar.gz
```
    
3. Move it to a convenient location:
```
mv wave-0.12.0-darwin-amd64 <pathname>/wave
```

4. check your $home/wave directory, the output should look like the following below:

```
.
├── demo
├── examples
├── readme.txt
├── test
├── waved
└── www
```
    
5. Go to your wave directory: 
    
```
cd wave
```
    
6. Start the wave server by running:

```
./waved
```
When runninng the command, you may get an issue about "cannot open file because identity of developer can't be verified:
        - go to system preferences
        - security & privacy
        - general 
        - you'll see a message about wave. select 'allow anyway'
        - attempt to run ./waved again

7. If step 6 worked you should be able to go to the following [link](http://localhost:10101/):
    
    - it's a spinning circle waiting for contact 
    - To run any Wave app, you need the Wave server up and running at all times.



### 2. Setup Your Python Environment

```bash
$ git clone https://github.com/h2oai/wave-amlb
$ cd wave-amlb
$ make setup
$ source venv/bin/activate
```

### 3. Run the App
After you set-up the python environment you can start the app by running

Note! this is in a separate terminal tab then where you ran './waved' command to start the wave server:
    
```bash
wave run src.app
```

Note! If you did not activate your virtual environment this will be:
    
```bash
./venv/bin/wave run src.app
```

### 4. View the App
    
Point your web browser to [localhost:10101](http://localhost:10101)
