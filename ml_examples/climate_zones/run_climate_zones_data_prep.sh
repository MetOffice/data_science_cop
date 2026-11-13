#!/bin/bash -l
#SBATCH --job-name=cz_data_prep
#SBATCH --output=cz_data_prep.out
#SBATCH --error=cz_data_prep.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#
# Convert the gridded climate zones netCDF data into ML-ready CSV files.
# Can be submitted to slurm or run directly with bash.
#
# usage - sbatch run_climate_zones_data_prep.sh /path/to/repo/ /path/to/output/dir/ [/path/to/source/data/]

set -euo pipefail

# -------- User-configurable arguments (passed to Python CLI) --------
export REPO_ROOT=$1
export OUTPUT_DIR=$2
export SCRIPT_DIR=${REPO_ROOT}/ml_examples/climate_zones
export CONFIG_PATH=${SCRIPT_DIR}/config.json

export RESOLUTIONS="1.0 0.5 0.1"
export NUM_PROCESSES=${SLURM_CPUS_PER_TASK:-4}
export FILE_FORMAT="nc"

# Source data dir defaults to the platform directory in config.json if not given.
ROOT_DATA_ARGS=()
if [ $# -ge 3 ]; then
  ROOT_DATA_ARGS=(--root-data-dir "$3")
fi

mkdir -p "${OUTPUT_DIR}"

# At the Met Office use `module load scitools/community/ml` instead.
conda activate ml-0.1.0

python "${SCRIPT_DIR}/climatezones_data_prep.py" \
  --config-path "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --resolutions ${RESOLUTIONS} \
  --num-processes "${NUM_PROCESSES}" \
  --file-format "${FILE_FORMAT}" \
  "${ROOT_DATA_ARGS[@]}"
