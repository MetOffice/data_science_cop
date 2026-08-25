#!/usr/bin/env bash
# This script is a small wrapper for intalling on fo the conda environments for the DS CoP on Azure ML. The wrapper
# primarily just calls conda create, and then uses pip to install the AzureML SDK and other packages that are
# not available through conda. It also sets up the kernel for use in Jupyter notebooks.

export CONDA_REQUIREMENTS_FILE=$1
export CONDA_ENV_NAME=$2

conda env create --file $CONDA_REQUIREMENTS_FILE

conda activate $CONDA_ENV_NAME
pip install mltable azure-ai-ml azureml-dataprep[pandas] azureml-fsspec azureml-mlflow

python -m ipykernel install --user --name $CONDA_ENV_NAME --display-name "$CONDA_ENV_NAME"