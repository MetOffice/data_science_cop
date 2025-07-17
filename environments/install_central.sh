#!/usr/bin/env bash
set -e

ENV_LIST="xgboost pytorch sklearn tensorflow tensorflow-gnn"

for env_name in ${ENV_LIST}; do
    echo creating ${env_name} environment
    conda env create --file environments/requirements_${env_name}.yml 
done
