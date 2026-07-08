#!/bin/bash -l
#SBATCH --job-name=climate-zones-torch
#SBATCH --output=logs/climate_zones_%j.out
#SBATCH --error=logs/climate_zones_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# -------- User-configurable arguments (passed to Python CLI) --------
export CONFIG_PATH="/home/users/stephen.haddad/prog/data_science_cop/ml_examples/climate_zones/config.json"
export LEARNING_RATE="0.001"
export BATCH_SIZE="16"
export NUM_EPOCHS="10"

# -------- MLflow server configuration --------
export MLFLOW_PORT="4455"
export MLFLOW_HOME=${DATADIR}/logs/climate_zones/

mkdir -p logs
mkdir -p ${MLFLOW_HOME}/artifacts

conda activate data-science-cop-torch

export MLFLOW_LOG_DIR=$SCRATCH/mlflow_logs
mkdir $MLFLOW_LOG_DIR

# Ensure MLflow runs from this job allocation
mlflow server \
  --backend-store-uri "sqlite:///${MLFLOW_HOME}/mlflow.db" \
  --default-artifact-root "${MLFLOW_HOME}/artifacts" \
  --host 127.0.0.1 \
  --port ${MLFLOW_PORT} \
  > "${MLFLOW_LOG_DIR}/mlflow_${SLURM_JOB_ID}.log" 2>&1 &

# Give MLflow server a moment to bind before training starts
sleep 8

SCRIPT_DIR="/home/users/stephen.haddad/prog/data_science_cop/ml_examples/climate_zones"
python "${SCRIPT_DIR}/ClimateZones_Training_Torch.py" \
  --config-path "${CONFIG_PATH}" \
  --learning-rate "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}"

