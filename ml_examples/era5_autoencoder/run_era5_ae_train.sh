#!/bin/bash -l
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=05:50:00
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --job-name=era5_ae_train_ai4c

set -e

export CONDA_ENV=/gws/ssde/j25a/mmh_storage/ai4c_conda/ai4c_cli_gpu
export MLFLOW_DIR=/gws/ssde/j25a/mmh_storage/user/shaddad/mlflow
export MLFLOW_PORT=4455

cd ~/prog/ai4c_hackathon/

# ./util/mlflow_server.sh  conda ${CONDA_ENV} ${MLFLOW_DIR} ${MLFLOW_PORT} &

export LEARNING_RATE=0.001
export BATCH_SIZE=16
export NUM_EPOCHS=10

conda activate ${CONDA_ENV}
# conda activate /gws/nopw/j04/mohc_shared/dscop/conda_envs/ai4c_hack_cli_gpu

python src/ai4c_hack/ERA5_autoencoder.py --config-path notebooks/config.json --model-out-dir  /gws/ssde/j25a/mmh_storage/user/shaddad/experiments/era5_autoencoder --batch-size=${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --learning-rate ${LEARNING_RATE} --data-dir /gws/ssde/j25a/mmh_storage/ai4c_data/weatherbench/mlready/norm/ # --mlflow-url "http://localhost" --mlflow-port ${MLFLOW_PORT}
