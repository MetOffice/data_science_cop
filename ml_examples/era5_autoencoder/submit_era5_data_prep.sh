#!/bin/bash -l

set -e

cd ~/prog/ai4c_hackathon/

export STD_OUT_PATH=/gws/nopw/j04/mohc_shared/users/shaddad/log/era5_data_prep_log_$(date '+%Y%m%d%H%M').out
export STD_ERR_PATH=/gws/nopw/j04/mohc_shared/users/shaddad/log/era5_data_prep_log_$(date '+%Y%m%d%H%M').err

sbatch -o $STD_OUT_PATH -e $STD_ERR_PATH util/run_era5_data_prep.sh 



