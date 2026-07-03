#!/bin/bash -l
# (C) British Crown Copyright 2017-2026, Met Office.
# Please see LICENSE.md for license details.
# This is a helper script for running ML Flow experiment tracker to log the details of your ML training runs
# If running on Jupyterhub on JASMIN:
# - This script should be run in a terminal on the notebook server and you should point to venv thast was setup for the notebook server.
# If running on JASMIN through command line:
# - This script should be run in a terminal on  server where you are doing the training. You should point to a conda environment for that server.
# Further information on running ML flow can be found in the docs  https://mlflow.org/docs/latest/ml/ 

export ENV_TOOL=${1:-venv} # can be venv or conda
export MLFLOW_ENV=${2:-~/venv/ai4c_nb_cpu}
export MLFLOW_ROOT=${3:-$HOME/mlflow}
export MLFLOW_PORT=${4:-4455}

if [ $ENV_TOOL = "conda" ]; then
    conda activate ${MLFLOW_ENV}
else
    . ${MLFLOW_ENV}/bin/activate
fi

echo starting MLflow server  "MLFLOW_ROOT: ${MLFLOW_ROOT}  MLFLOW_PORT: ${MLFLOW_PORT}"
mlflow server --backend-store-uri "sqlite:////${MLFLOW_ROOT}/backend.db" --artifacts-destination ${MLFLOW_ROOT}/artifacts/ --registry-store-uri ${MLFLOW_ROOT}/registry/ --port ${MLFLOW_PORT}
