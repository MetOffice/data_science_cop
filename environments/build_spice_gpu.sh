#!/bin/bash -l
#SBATCH --mem=32G
#SBATCH --ntasks=4
#SBATCH --time=60
#SBATCH --export=NONE
#SBATCH --job-name=conda_build_spice
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

echo $1
export REQUIREMENTS_FILE=$1
export ENV_NAME=$2

conda env create --file $REQUIREMENTS_FILE --prefix /data/users/stephen.haddad/conda/${ENV_NAME}

