#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --account=mohc_shared
#SBATCH --qos=high
#SBATCH --time=02:00:00
#SBATCH --ntasks=4
#SBATCH --mem=64G
#SBATCH --job-name=ai4c_era5_data_prep

set -e

conda activate /gws/nopw/j04/mohc_shared/dscop/conda_envs/ai4c_hack_cli_cpu

cd ~/prog/ai4c_hackathon/

python src/ai4c_hack/ERA5_data_prep.py --start-year 1980 --end-year 2016 --data-out-dir  /gws/ssde/j25a/mmh_storage/ai4c_data/weatherbench/mlready --config notebooks/config.json



