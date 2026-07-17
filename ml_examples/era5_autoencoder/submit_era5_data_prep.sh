#!/bin/bash -l

set -e

cd ~/prog/data_science_cop/ml_examples/era5_autoencoder/

# change to point to the directory where you store your log files
export LOG_DIR=/gws/ssde/j25b/mohc_shared/users/shaddad/log/

export STD_OUT_PATH=$LOG_DIR/era5_data_prep_log_$(date '+%Y%m%d%H%M').out
export STD_ERR_PATH=$LOG_DIR/era5_data_prep_log_$(date '+%Y%m%d%H%M').err

sbatch -o $STD_OUT_PATH -e $STD_ERR_PATH run_era5_data_prep.sh 



