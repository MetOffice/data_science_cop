#!/bin/bash -l
set -e

# Get command line arguments
export REPO_ROOT=$1
export USER_LOG_DIR=$2
export EXPERIMENT_INPUT_DIR=$3
export EXPERIMENT_OUTPUT_DIR=$4
export CONDA_ENV=$5


# This script need to run from the root directory of the data science cop repository, so update the path below accordingly
cd ${REPO_ROOT}/ml_examples/era5_autoencoder/

# Set to a suitable directory for writing log files (for which you have write permission).
export USER_DIR=/gws/ssde/j25b/mohc_shared/users/$USER

#uncomment these lines if you do not have a user directory yet
# mkdir $USER_DIR
# mkdir $USER_LOG_DIR


export STD_OUT_PATH=${USER_LOG_DIR}/era5_ae_train_log_$(date '+%Y%m%d%H%M').out
export STD_ERR_PATH=${USER_LOG_DIR}/era5_ae_train_log_$(date '+%Y%m%d%H%M').err
export JOBNAME=era5_autoencoder_train_$(date '+%Y%m%d%H%M')

echo Writing logs to $STD_OUT_PATH and $STD_ERR_PATH

sbatch -o $STD_OUT_PATH -e $STD_ERR_PATH -J ${JOBNAME} run_era5_ae_train_spice.sh ${REPO_ROOT} ${USER_LOG_DIR} ${EXPERIMENT_INPUT_DIR} ${EXPERIMENT_OUTPUT_DIR} ${CONDA_ENV}



