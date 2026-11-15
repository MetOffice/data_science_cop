#!/bin/bash -l
#SBATCH --job-name=cz_eval
#SBATCH --output=cz_eval.out
#SBATCH --error=cz_eval.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# Evaluate the PyTorch climate zones model trained by climatezones_training_torch.py
# on the train/val/test splits.
#
# The model and stats.json are written to the MLflow artifacts of the training run,
# e.g. ${LOG_DIR}/artifacts/<experiment_id>/<run_id>/artifacts/
#
# usage - sbatch run_climate_zones_evaluation.sh /path/to/repo/ /path/to/output/dir/ /path/to/training/run/artifacts/

set -euo pipefail

# -------- User-configurable arguments (passed to Python CLI) --------
export REPO_ROOT=$1
export OUTPUT_DIR=$2
export ARTIFACTS_DIR=$3
export SCRIPT_DIR=${REPO_ROOT}/ml_examples/climate_zones
export CONFIG_PATH=${SCRIPT_DIR}/config.json
export MODEL_PATH=${ARTIFACTS_DIR}/cz_model.pth
export STATS_PATH=${ARTIFACTS_DIR}/stats.json

export ALGORITHM_NAME="torch_nn"
export RESOLUTION="1.0"
export TARGET_VAR="climate_group"
export TEST_FRAC="0.1"
export VAL_FRAC="0.1"
export BATCH_SIZE="65536"
export DEVICE="cpu"

mkdir -p "${OUTPUT_DIR}"

conda activate dscop_pytorch

python "${SCRIPT_DIR}/climatezones_evaluation.py" \
  --config-path "${CONFIG_PATH}" \
  --stats-path "${STATS_PATH}" \
  --model-path "${MODEL_PATH}" \
  --algorithm-name "${ALGORITHM_NAME}" \
  --resolution "${RESOLUTION}" \
  --target-var "${TARGET_VAR}" \
  --test-frac "${TEST_FRAC}" \
  --val-frac "${VAL_FRAC}" \
  --batch-size "${BATCH_SIZE}" \
  --device "${DEVICE}" \
  --output-dir "${OUTPUT_DIR}"
