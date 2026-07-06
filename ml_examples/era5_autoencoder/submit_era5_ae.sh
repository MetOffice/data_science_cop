#!/bin/bash -l
set -e

cd ~/prog/ai4c_hackathon/

export STD_OUT_PATH=/gws/ssde/j25a/mmh_storage/user/shaddad/log/era5_ae_train_log_$(date '+%Y%m%d%H%M').out
export STD_ERR_PATH=/gws/ssde/j25a/mmh_storage/user/shaddad/log/era5_ae_train_log_$(date '+%Y%m%d%H%M').err

sbatch -o $STD_OUT_PATH -e $STD_ERR_PATH util/run_era5_ae_train.sh 



