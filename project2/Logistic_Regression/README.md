# Story Cloze Test - NLU Project 2

# Virtual environment 

Create a new conda virtual environment with required packages

```
conda create --name nlu_env_lr python=3.6.4 -y
```

Activate the virtual environment

```
source activate nlu_env_lr
```

Install pip requirements

```
pip install -r requirements.txt
```

# Running

In case the code is being run on a cluster, run the following before submitting a job:

```
python3 -m nltk.downloader vader_lexicon
python3 -m nltk.downloader averaged_perceptron_tagger
python3 -m nltk.downloader punkt
```

The main code is executed by the command:

```
python3 Logistic_Regression.py
```

Two prediction files are generated: predictions_s.csv and predctions_es.csv where, the former is the prediction considering only sentiment features while the later is the prediciton considering both sentiment and event features.
