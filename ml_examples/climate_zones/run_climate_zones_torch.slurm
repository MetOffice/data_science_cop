#!/bin/bash -l
#SBATCH --job-name=cz_train
#SBATCH --output=cz_train.out
#SBATCH --error=cz_train.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# usage -  sbatch run_climate_zones_torch.slurm /path/to/repo/ /path/to/data/output/dir/\

set -euo pipefail

# -------- User-configurable arguments (passed to Python CLI) --------
export REPO_ROOT=$1
export CONFIG_PATH=${REPO_ROOT}/ml_examples/climate_zones/config.json
export SCRIPT_DIR=${REPO_ROOT}/ml_examples/climate_zones

export LEARNING_RATE="0.001"
export BATCH_SIZE="16"
export NUM_EPOCHS="5"

export LOG_DIR=$2

# -------- MLflow server configuration --------
export MLFLOW_PORT="4455"
export MLFLOW_HOME=${LOG_DIR}
export MLFLOW_ARTIFACTS_DIR=${MLFLOW_HOME}/artifacts
if [ ! -d ${MLFLOW_ARTIFACTS_DIR} ];then
  mkdir -p ${MLFLOW_ARTIFACTS_DIR}
fi
export MLFLOW_LOG_DIR=$SCRATCH/mlflow_logs
if [ ! -d ${MLFLOW_LOG_DIR} ];then
  mkdir $MLFLOW_LOG_DIR
fi

conda activate dscop_pytorch

# Ensure MLflo  w runs from this job allocation
mlflow server \
  --backend-store-uri "sqlite:///${MLFLOW_HOME}/mlflow.db" \
  --default-artifact-root "${MLFLOW_ARTIFACTS_DIR}" \
  --host 127.0.0.1 \
  --port ${MLFLOW_PORT} \
  > "${MLFLOW_LOG_DIR}/mlflow_${SLURM_JOB_ID}.log" 2>&1 &

# Give MLflow server a moment to bind before training starts
sleep 8

python "${SCRIPT_DIR}/ClimateZones_Training_Torch.py" \
  --config-path "${CONFIG_PATH}" \
  --learning-rate "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --output-dir "${MLFLOW_HOME}" \
  --start-epoch-num 10 \
  --starting-checkpoint /data/users/stephen.haddad/logs/climate_zones/artifacts/1/bb1b28d1ceff4130b84a0a529e52d270/artifacts/cz_model.pth


