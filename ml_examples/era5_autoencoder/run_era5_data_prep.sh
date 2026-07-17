#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --account=mohc_shared
#SBATCH --qos=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --job-name=dscop_era5_ae_data_prep

set -e

export CONDA_ENV=/gws/ssde/j25b/mohc_shared/dscop/conda_envs/dscop_pytorch_cpu
export DATA_OUTPUT_DIR=/gws/ssde/j25b/mohc_shared/dscop/weatherbench/mlready/

conda activate ${CONDA_ENV}
cd ~/prog/data_science_cop

python ml_examples/era5_autoencoder/era5_autoencoder_data_prep.py --start-year 1980 --end-year 2016 --data-out-dir  $DATA_OUTPUT_DIR --root-data-dir /gws/ssde/j25a/mmh_storage/ai4c_data/weatherbench/



