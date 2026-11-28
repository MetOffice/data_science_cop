#!/usr/bin/env bash
set -e

ENV_LIST="xgboost pytorch sklearn tensorflow tfgnn"
CONDA_DIR=$DATADIR/conda
for env_name in ${ENV_LIST}; do
    echo creating ${env_name} environment
    conda create --file environments/requirements_${env_name}_molinux.lock --prefix ${CONDA_DIR}/dscop_${env_name}
done
