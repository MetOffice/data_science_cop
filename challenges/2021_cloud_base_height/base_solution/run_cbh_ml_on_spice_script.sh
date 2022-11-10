#!/bin/bash -l
#SBATCH --mem=10G
#SBATCH --ntasks=2
#SBATCH --output=cbh_ml_spice_output.txt
#SBATCH --time=1440
#SBATCH --export=NONE
#SBATCH --qos=long
module load scitools/experimental-current
/net/home/h01/frme/cyrilmorcrette-projects/python/cbh_ml.py
