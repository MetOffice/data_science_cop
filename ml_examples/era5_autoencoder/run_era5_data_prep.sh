#!/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --job-name=dscop_era5_ae_data_prep

set -e

export REPO_ROOT=$1
export LOG_DIR=$2
export DATA_INPUT_DIR=$3
export DATA_OUTPUT_DIR=$4
export CONDA_ENV=$5


conda activate ${CONDA_ENV}
cd ${REPO_ROOT}
cd ml_examples/era5_autoencoder/

python era5_autoencoder_data_prep.py --start-year 1980 --end-year 2016 --data-out-dir  $DATA_OUTPUT_DIR --root-data-dir $DATA_INPUT_DIR



