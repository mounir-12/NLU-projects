#!/bin/bash

cd ~/nlu

module purge
module load gcc/6.3.0
module load python_gpu/3.7.1

bsub -n 8 -W 24:00 -o output_pretrain -R "rusage[mem=8192, ngpus_excl_p=1]" "python run_pretraining.py --input_file=data/bert_corpus.tfrecord --output_dir=data/pretraining_output --do_train=True --do_eval=True --bert_config_file=data/models/bert/uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=data/models/bert/uncased_L-12_H-768_A-12/bert_model.ckpt --train_batch_size=16 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=10000 --num_warmup_steps=1000 --learning_rate=2e-5"