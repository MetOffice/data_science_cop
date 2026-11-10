#!/bin/bash -l
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=08:50:00
#SBATCH --ntasks=16
#SBATCH --mem=64G
#SBATCH --job-name=dscop_era5_ae_train

set -e


# user directories
export USER_DIR=/gws/ssde/j25b/mohc_shared//users/$USER
export USER_EXP_DIR=$USER_DIR/experiments

#uncomment these lines if you do not have a user directory yet
# mkdir $USER_DIR
# mkdir $USER_EXP_DIR

export CONDA_ENV=/gws/ssde/j25b/mohc_shared/dscop/conda_envs/dscop_pytorch_gpu/
export MLFLOW_DIR=${USER_DIR}/mlflow
export MLFLOW_PORT=4455

cd ~/prog/data_science_cop
cd ml_examples/era5_autoencoder/

export WEATHERBENCH_NORM_DIR=/gws/ssde/j25a/mmh_storage/ai4c_data/weatherbench/mlready/norm/

# ./util/mlflow_server.sh  conda ${CONDA_ENV} ${MLFLOW_DIR} ${MLFLOW_PORT} &

export LEARNING_RATE=0.001
export BATCH_SIZE=16
export NUM_EPOCHS=10

conda activate ${CONDA_ENV}

python train_era5_autoencoder.py --config-path config.json --model-out-dir ${USER_EXP_DIR} --batch-size=${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --learning-rate ${LEARNING_RATE} --data-dir $WEATHERBENCH_NORM_DIR  # --mlflow-url "http://localhost" --mlflow-port ${MLFLOW_PORT}

echo cleaning cache dir $DATA_CACHE_DIR
rm -rf $DATA_CACHE_DIR
