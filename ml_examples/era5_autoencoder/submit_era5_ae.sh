#!/bin/bash -l
set -e

# This script need to run from the root directory of the data science cop repository, so update the path below accordingly
cd ~/prog/data_science_cop

cd ml_examples/era5_autoencoder/

# Set to a suitable directory for writing log files (for which you have write permission).
export LOG_DIR=/gws/ssde/j25a/mmh_storage/user/shaddad/log/

export STD_OUT_PATH=${LOG_DIR}/era5_ae_train_log_$(date '+%Y%m%d%H%M').out
export STD_ERR_PATH=${LOG_DIR}/era5_ae_train_log_$(date '+%Y%m%d%H%M').err

sbatch -o $STD_OUT_PATH -e $STD_ERR_PATH run_era5_ae_train.sh 



