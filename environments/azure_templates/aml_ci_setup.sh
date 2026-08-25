#!/bin/bash -l
# This script is to run on a new compute instance to set it up for the DS CoP AzureML tutorials. It should be run on the compute instance.

export PROG_ROOT_DIR=$PWD

export ML_WEATHER_TUTORIAL_DIR=$PROG_ROOT_DIR/ml_weather_tutorial
git clone https://github.com/MetOffice/ml_weather_tutorial/ ${ML_WEATHER_TUTORIAL_DIR}
git config --global --add safe.directory ${ML_WEATHER_TUTORIAL_DIR}

export DSCOP_REPO_DIR=$PROG_ROOT_DIR/data_science_cop
git clone https://github.com/MetOffice/data_science_cop.git ${DSCOP_REPO_DIR}
git config --global --add safe.directory ${DSCOP_REPO_DIR}

# this is temporary until we can get the azure_pathways branch merged into main
cd $PROG_ROOT_DIR/data_science_cop
git switch azure_pathways

cd environments

# conda doesn't seem to work until you accept the terms of service so adding these commands
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda env create --file requirements_pytorch.yml
conda activate dscop_pytorch
pip install mltable azure-ai-ml azureml-dataprep[pandas]
python -m ipykernel install --user --name dscop_pytorch --display-name "dscop_pytorch_azureml"



