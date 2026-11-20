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
export TRAINING_DATA_DIR=$2
export MODEL_OUTPUT_DIR=$3
export EVALUATION_DIR=$4
export CONDA_ENV=$5

export MODEL_FNAME=${MODEL_FNAME:-"cz_model.pth"}

export SCRIPT_DIR=${REPO_ROOT}/ml_examples/climate_zones
export CONFIG_PATH=${SCRIPT_DIR}/config.json
export MODEL_PATH=${MODEL_OUTPUT_DIR}/${MODEL_FNAME}
echo loading model from ${MODEL_PATH}
export STATS_PATH=${MODEL_OUTPUT_DIR}/stats.json

export RESOLUTION="1.0"

mkdir -p "${EVALUATION_DIR}"

conda activate "${CONDA_ENV}"

python "${SCRIPT_DIR}/climatezones_evaluation.py" \
  --config-path "${CONFIG_PATH}" \
  --stats-path "${STATS_PATH}" \
  --model-path "${MODEL_PATH}" \
  --resolution "${RESOLUTION}" \
  --training-data-dir "${TRAINING_DATA_DIR}" \
  --evaluation-dir "${EVALUATION_DIR}"


