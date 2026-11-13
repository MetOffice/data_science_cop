#!/bin/bash -l
#SBATCH --job-name=cz_eval
#SBATCH --output=cz_eval.out
#SBATCH --error=cz_eval.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#
# Evaluate a trained climate zones model on the train/val/test splits.
#
# usage - sbatch run_climate_zones_evaluation.sh /path/to/repo/ /path/to/output/dir/ [/path/to/model.pkl] [/path/to/stats.json]

set -euo pipefail

# -------- User-configurable arguments (passed to Python CLI) --------
export REPO_ROOT=$1
export OUTPUT_DIR=$2
export SCRIPT_DIR=${REPO_ROOT}/ml_examples/climate_zones
export CONFIG_PATH=${SCRIPT_DIR}/config.json
export MODEL_PATH=${3:-${SCRIPT_DIR}/decision_tree.pkl}
export STATS_PATH=${4:-${SCRIPT_DIR}/stats.json}

export ALGORITHM_NAME="decision_tree"
export RESOLUTION="1.0"
export TARGET_VAR="climate_group"
export TEST_FRAC="0.1"
export VAL_FRAC="0.1"

mkdir -p "${OUTPUT_DIR}"

# At the Met Office use `module load scitools/community/ml` instead.
conda activate ml-0.1.0

python "${SCRIPT_DIR}/climatezones_evaluation.py" \
  --config-path "${CONFIG_PATH}" \
  --stats-path "${STATS_PATH}" \
  --model-path "${MODEL_PATH}" \
  --algorithm-name "${ALGORITHM_NAME}" \
  --resolution "${RESOLUTION}" \
  --target-var "${TARGET_VAR}" \
  --test-frac "${TEST_FRAC}" \
  --val-frac "${VAL_FRAC}" \
  --output-dir "${OUTPUT_DIR}"
# To train a fresh decision tree instead of loading MODEL_PATH, add:
#  --train-new-model --max-depth 10 --min-samples-leaf 2 --min-samples-split 5
