# dsta-smart-search
BT4103 Capstone Project

# Setup:

1. Create an anaconda enviornment with python=3.6
1. Update your environment with `conda env update -n <environment name> -f environment.yml --prune`
1. Run main.py


# Structure of this folder (folders/files not mentioned are redundant):

1. `data/train-v2.0.json` contains the input data for SQuAD 2.0. The 443 categories, along with all the documents are in this file.
1. `Dockerfile` contains the instruction for Docker to build the image.
1. `environment.yml` contains the anaconda environment that was used to build this environment .
1. `main.py` contains the code for this entire application.
1. `requirements.txt` contains the pip packages that were used for this application. Every package in this file will be downloaded by docker. Otherwise, you could upgrade your environment with `pip install -r requirements.txt` as well.