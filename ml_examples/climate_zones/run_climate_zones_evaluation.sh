#!/bin/bash -l
#SBATCH --job-name=cz_eval
#SBATCH --output=/data/users/%u/logs/%x.out
#SBATCH --error=/data/users/%u/logs/%x.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=4
#SBATCH --mem=16G
#
# Evaluate the PyTorch climate zones model trained by climatezones_training_torch.py
# on the train/val/test splits.
#
# The model and stats.json are written to the MLflow artifacts of the training run,
# e.g. ${LOG_DIR}/artifacts/<experiment_id>/<run_id>/artifacts/
#
# usage - sbatch run_climate_zones_evaluation.sh /path/to/repo/ /path/to/output/dir/ /path/to/training/run/artifacts/ conda_env_name

set -euo pipefail

# -------- User-configurable arguments (passed to Python CLI) --------
export REPO_ROOT=$1
export OUTPUT_DIR=$2
export ARTIFACTS_DIR=$3
export CONDA_ENV=$4

export SCRIPT_DIR=${REPO_ROOT}/ml_examples/climate_zones
export CONFIG_PATH=${SCRIPT_DIR}/config.json
export MODEL_PATH=${ARTIFACTS_DIR}/cz_model.pth
export STATS_PATH=${ARTIFACTS_DIR}/stats.json

export RESOLUTION="1.0"

mkdir -p "${OUTPUT_DIR}"

conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/climatezones_evaluation.py" \
  --config-path "${CONFIG_PATH}" \
  --stats-path "${STATS_PATH}" \
  --model-path "${MODEL_PATH}" \
  --resolution "${RESOLUTION}" \
  --output-dir "${OUTPUT_DIR}"


