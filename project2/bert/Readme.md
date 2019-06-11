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

Install pip requirements

```
pip install -r requirements.txt
```

### Data
Please put all the project's csv files under `./data/`
The provided preprocessed files are:
- training set: val.csv
- test and predictions set: test.csv

### BERT
https://arxiv.org/abs/1810.04805

Idea: Use the fourth sentences as "sentence a" and the last as "sentence b", do binary classification on this input

### Preprocess data
All the data is preprocessed and generated from the "data" notebook. 

### Run

#### Pre-training
You can run the pre-training on the training data set using the following command:
```
python run_pretraining.py 
    \ --input_file=data/bert_corpus.tfrecord -
    \ -output_dir=data/pretraining50k_output 
    \ --do_train=True 
    \ --do_eval=True 
    \ --bert_config_file=data/models/bert/uncased_L-12_H-768_A-12/bert_config.json 
    \ --init_checkpoint=data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt 
    \ --train_batch_size=16 --max_seq_length=128 --max_predictions_per_seq=20 
    \ --num_train_steps=50000 --num_warmup_steps=5000 --learning_rate=2e-5
```

The code to generate the `bert_corpus.tfrecord` file is inside the Jupyter notebook.

#### Finetuning
We only use the validation data for finetuning. The file `bert_model.py` can be used to train a model and predict on the prediction set.

To **train, validate, test and predict**, simply run:
```
python3 bert_model.py
```

After the training is completed, the model will be evaluated on the test set
