#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --job-name=dscop_era5_ae_train

set -e



# Get command line arguments
export REPO_ROOT=$1
export USER_LOG_DIR=$2
export EXPERIMENT_INPUT_DIR=$3
export USER_EXP_DIR=$4
export CONDA_ENV=$5

# user directories
export USER_DIR=/gws/ssde/j25b/mohc_shared//users/$USER

# export MLFLOW_DIR=${USER_LOG_DIR}/mlflow
# export MLFLOW_PORT=4455

cd ${REPO_ROOT}/ml_examples/era5_autoencoder/

# ./util/mlflow_server.sh  conda ${CONDA_ENV} ${MLFLOW_DIR} ${MLFLOW_PORT} &

export LEARNING_RATE=0.001
export BATCH_SIZE=16
export NUM_EPOCHS=10

conda activate ${CONDA_ENV}

python train_era5_autoencoder.py --config-path config.json --model-out-dir ${USER_EXP_DIR} --batch-size=${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --learning-rate ${LEARNING_RATE} --data-dir ${EXPERIMENT_INPUT_DIR}  # --mlflow-url "http://localhost" --mlflow-port ${MLFLOW_PORT}
