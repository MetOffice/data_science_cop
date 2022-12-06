#!/bin/bash -l
#SBATCH --mem=150G
#SBATCH --ntasks=24
#SBATCH --time=
conda activate py-lightning
python 03_pytorch_train.py > slurm_python_log.txt
