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
The required data files files are:
- training files:
  - `ROCStories__spring2016 - ROCStories_spring2016.csv`
  - `ROCStories_winter2017 - ROCStories_winter2017.csv`
- validation file: `cloze_test_val__spring2016 - cloze_test_ALL_val.csv`
- test file: `cloze_test_test__spring2016 - cloze_test_ALL_test.csv`
- prediction file: `test-stories.csv`

As you can see, we collected additional training data from the Story Cloze website: http://cs.rochester.edu/nlp/rocstories/

You'll also need the base BERT model which you can download [here](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip).
Please unzip it under `./data/models/bert/uncased_L-12_H-768_A-12/`.

### BERT

https://github.com/google-research/bert

### Run

#### Preprocess data
All the data is preprocessed and generated from the `data.ipynb` notebook.
This will allow you to get the files required later for the training process:
- `bert_corpus.tfrecord`
- `val.csv`
- `test.csv`

Those will be generated under `./data/`.

If you want to play it quick, you can run the script version:
```
python3 data.py
```

To generate the `predict.csv` file required for predictions on the unlabeled data, run the following:
```
python3 create_pretraining_data.py
```
 
#### Pre-training
You can run the pre-training on the training data set using the following command:
```
python3 run_pretraining.py 
    \ --input_file=data/bert_corpus.tfrecord
    \ -output_dir=data/pretraining50k_output 
    \ --do_train=True 
    \ --do_eval=True 
    \ --bert_config_file=data/models/bert/uncased_L-12_H-768_A-12/bert_config.json 
    \ --init_checkpoint=data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt 
    \ --train_batch_size=16 --max_seq_length=128 --max_predictions_per_seq=20 
    \ --num_train_steps=50000 --num_warmup_steps=5000 --learning_rate=2e-5
```

#### Finetuning
We only use the validation data for finetuning. The file `bert_model.py` can be used to train a model and predict on the prediction set.

To **train, validate, test and predict**, simply run:
```
python3 bert_model.py
```

After the training is completed, the model will be evaluated on the test set and predictions will be run on the appropriate file.

## Model files
Because they can be easily generated, the model files have not been provided.
However, the authors of this work are ready to provide them free of charge if you want to avoid the hassle of running anything. Please ask in case of need.