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

cd ~/prog/data_science_cop
cd ml_examples/era5_autoencoder/

export WEATHERBENCH_NORM_DIR=/gws/ssde/j25a/mmh_storage/ai4c_data/weatherbench/mlready/norm/
export DATA_CACHE_DIR=/tmp/era5_autoencoder/
mkdir $DATA_CACHE_DIR

echo copying data to $DATA_CACHE_DIR
cp -r $WEATHERBENCH_NORM_DIR/* $DATA_CACHE_DIR/

# ./util/mlflow_server.sh  conda ${CONDA_ENV} ${MLFLOW_DIR} ${MLFLOW_PORT} &

export LEARNING_RATE=0.001
export BATCH_SIZE=16
export NUM_EPOCHS=10

conda activate ${CONDA_ENV}

python ERA5_autoencoder.py --config-path config.json --model-out-dir  /gws/ssde/j25a/mmh_storage/user/shaddad/experiments/era5_autoencoder --batch-size=${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --learning-rate ${LEARNING_RATE} --data-dir $DATA_CACHE_DIR  # --mlflow-url "http://localhost" --mlflow-port ${MLFLOW_PORT}

echo cleaning cache dir $DATA_CACHE_DIR
rm -rf $DATA_CACHE_DIR
