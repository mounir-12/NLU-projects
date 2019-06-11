#!/bin/bash

cd ~/nlu

module purge
module load gcc/6.3.0
module load python_gpu/3.7.1

bsub -n 8 -W 12:00 -o output2 -R "rusage[mem=4096, ngpus_excl_p=1]"  "python3 bert_model.py"