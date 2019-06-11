#!/bin/bash

cd ~/nlu

module purge
module load gcc/6.3.0
module load python_gpu/3.7.1

bsub -n 2 -W 00:30 -o eval1.out -R "rusage[mem=2048, ngpus_excl_p=1]"  "python3 bert_model.py -op eval1/model -m eval predict $1"