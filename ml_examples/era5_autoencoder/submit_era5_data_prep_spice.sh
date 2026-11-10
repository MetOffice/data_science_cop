#!/bin/bash -l

set -e

export REPO_ROOT=$1
export LOG_DIR=$2
export DATA_INPUT_DIR=$3
export DATA_OUT_DIR=$4
export CONDA_ENV=$5

export STD_OUT_PATH=$LOG_DIR/era5_data_prep_log_$(date '+%Y%m%d%H%M').out
export STD_ERR_PATH=$LOG_DIR/era5_data_prep_log_$(date '+%Y%m%d%H%M').err

cd $REPO_ROOT/ml_examples/era5_autoencoder/

sbatch -o $STD_OUT_PATH -e $STD_ERR_PATH run_era5_data_prep_spice.sh $REPO_ROOT ${LOG_DIR} ${DATA_INPUT_DIR} ${DATA_OUT_DIR} ${CONDA_ENV}
