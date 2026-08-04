#!/bin/bash
# This script is to run on a new compute instance to set it up for the DS CoP AzureML tutorials. It should be run on the compute instance.
export DATA_ROOT_DIR=~/data

mkdir $DATA_ROOT_DIR
cd $DATA_ROOT_DIR
wget https://zenodo.org/records/7390654/files/xbt_1968.csv
wget https://zenodo.org/records/21773439/files/climate_zones_1p0.csv

mkdir ~/downloads/
# cd ~/downloads
# bash ./Miniconda3-latest-Linux-x86_64.sh

export PROG_ROOT_DIR=~/prog
mkdir $PROG_ROOT_DIR

git clone https://github.com/informatics-lab/ml_weather_tutorial/ $PROG_ROOT_DIR/ml_weather_tutorial
git clone https://github.com/MetOffice/data_science_cop.git $PROG_ROOT_DIR/data_science_cop

# this is temporary until we can get the azure_pathways branch merged into main
cd $PROG_ROOT_DIR/data_science_cop
git switch azure_pathways





