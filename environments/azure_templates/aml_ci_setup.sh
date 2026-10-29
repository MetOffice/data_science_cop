#!/bin/bash -l
# This script is to run on a new compute instance to set it up for the DS CoP AzureML tutorials. It should be run on the compute instance.
set -e

#the script should be run fomm the user directory, which is the same one that it is copied to in the workspace set up script
export PROG_ROOT_DIR=$PWD

# copy the conda and artifactory setup files into place
#cp condarc ~/.condarc
cp netrc ~/.netrc

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

## you need to copy across your .etc file for this to work
touch ~/.condarc
printf "channel_alias: https://metoffice.jfrog.io/metoffice/api/conda
channels:
  - conda-mo-internal
  - conda-forge
" > ~/.condarc

conda init

./azure_templates/azure_conda_setup.sh $PWD/azure_templates/requirements_pytorch_aml.yml dscop_pytorch_aml



