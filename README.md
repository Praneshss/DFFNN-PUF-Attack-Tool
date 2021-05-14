# DFFNN-PUF-Attack-Tool
Deep FeedForward Neural Network based Model Building Attack Tool on PUF

### Prerequisites

What things you need to install the software and how to install them

```
python >= 3.6
tensorflow >= 2.0
kerastuner


Other important packages required in python:
numpy 
pandas
sklearn
```

## Running the tests

For running the tests, go to the respective PUF folder and execute
```
python dfnn_attack.py --challenge=<path>XOR_APUF_Binary_chal_64_500000.csv  --response=<path>/4-xorpuf.csv  --features=parity --level=1 --verbose=0

For more options:
python dfnn_attack.py -h
usage: dfnn_attack.py [-h] [-c path] [-r path] [-f FEATURES] [-l LEVEL]
                      [-v VERBOSE]

Deep learning based model building attack tool on PUF

optional arguments:
  -h, --help            show this help message and exit
  -c path, --challenge path
                        the path to challenges CSV file
  -r path, --response path
                        the path to response CSV file
  -f FEATURES, --features FEATURES
                        To convert the challenges to feature vectors
  -l LEVEL, --level LEVEL
                        level specifies the first-level test (0.5 million
                        CRPs) or second level test ( <= 1.5 million)
  -v VERBOSE, --verbose VERBOSE
                        verbose=10 enables the epoch prints and verbose=0
                        disables it


```
