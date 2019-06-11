# Story Cloze Test - NLU Project 2

## Getting Started

### Virtual environment

Create a new conda virtual environment with required packages

```
conda create --name nlu_env_sa python=3.6.4 -y
```

Activate the virtual environment

```
source activate nlu_env_sa
```

Install mkl

```
conda install mkl-service -y
```

Install pip requirements

```
pip install -r requirements.txt
```
### Pre-trained skip-thought embeddings

The model uses pre-trained skip-thoughts embeddings and the modified file `skipthoughts.py` from [ryankiros](https://github.com/ryankiros/skip-thoughts). You need to download the 
the skip-thoughts model files and word embeddings, please go to `./models/` and run:
```
./download_files.sh
```

### Data
Please put all the project's csv files under `./data/`
The provided files are:
- training set: train_stories.csv
- validation set: cloze_test_val__spring2016 - cloze_test_ALL_val.csv
- test set: test_for_report-stories_labels.csv
- predictions set: test-stories.csv

### Preprocess data
You need to start by encoding the validation, test and prediction sentences. You can do this by running:

```
python3 preprocess_data.py
```
A new folder called `./encoded_data/` is created with the encoded sentences of the validation, test and prediction sets stored as h5 files.
These files with encoded sentences are used by the training code, so this step is necessary.

In case the original data is changed, please remove the folder `./encoded_data/` and rerun the code in `preprocess_data.py`

## Running
This model only uses the labeled validation data for training. We train on 90% of this data and validate on the remaining 10%
The file `train_predict_sa.py` can be used to train a model and predict on the prediction set.
Every epoch, the train, validation and test accuracies are computed and the model's new weights are saved if the model achieves better validation accuracy than in the previous epochs.
After training, the best model weights are reloaded and prediction is done using these weights.

To **train, validate, test and predict**, simply run:
```
python3 train_predict_sa.py
```
To skip training and only **predict** using the best model weight of the last trained model, run:
```
python3 train_predict_sa.py -st
```
The predictions file is generated in the model folder created under `./logs/`. The model weights and some tensorboard files are saved in that same folder.

Other possible options:
```
python3 train_predict_sa.py -e <nb_epochs> -bs <batch_size> -d <dropout_rate> -lr <learning_rate>
```
Use the option -st to skip training
Use the option -s to use as input to the network the sum of the last contex sentence with the story ending instead of their concatenation
Use the option -v for some more verbosity (used for dev)

