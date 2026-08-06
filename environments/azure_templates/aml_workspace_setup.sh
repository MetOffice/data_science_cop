#!/usr/bin/env bash
# This script is to be run from any machine that
# has the Azure CLI installed and configured to create an AzureML workspace
# and associated resources for the DS CoP AzureML tutorials.

# Login to AzureML CLI - If your terminal is not currently running in a logged-in Azure CLI session, run the following command to log in:
# az login

#use this if on Met Office IT
export PLATFORM=metoffice

# use for other
# export PLATFORM=local


export WORKSPACE_NAME=dscoptest1
export RESOURCE_GROUP=1-1a0e0d26-playground-sandbox

# this must be updated by the user before running
export USER_NAME=cloud_user_p_dcc359dd/
#===============================
# Create workspace

python create_workspace_spec.py --name $WORKSPACE_NAME --resource-group $RESOURCE_GROUP
az ml workspace create --file workspace_spec.yaml --resource-group $RESOURCE_GROUP


#===============================
# Create data stores and data assets

if [ "${PLATFORM}" = "metoffice" ]; then
  export XBT_DATA_PATH=/data/users/dscop/ml_tutorial/xbt/xbt_1968.csv
  export CLIMATE_ZONES_DATA_PATH=/data/users/dscop/ml_tutorial/climate_zones/ml_ready/climate_zones_1p0.csv
else
  export XBT_DATA_PATH=xbt_1968.csv
  export CLIMATE_ZONES_DATA_PATH=climate_zones_1p0.csv
  wget -O "${XBT_DATA_PATH}" https://zenodo.org/records/7390654/files/xbt_1968.csv
  wget -O "${CLIMATE_ZONES_DATA_PATH}" https://zenodo.org/records/21773439/files/climate_zones_1p0.csv
fi

azcopy copy "${XBT_DATA_PATH}" "${CONTAINER_URL}/xbt/xbt_1968.csv"
azcopy copy "${CLIMATE_ZONES_DATA_PATH}" "${CONTAINER_URL}/climate_zones/climate_zones_1p0.csv"
# create workspace

export STORAGE_ACCOUNT=$(python get_storage_account.py $RESOURCE_GROUP --first-only)

export DSCOP_DATASTORE_NAME=dscopworkspacestore
python create_workspace_spec.py datastore --name $DSCOP_DATASTORE_NAME --account-name $STORAGE_ACCOUNT --container-name $CONTAINER_NAME --description="Main datastore for blob storage associated with workspace."

az ml datastore create --file datastore_spec.yaml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME


export XBT_AML_URI="azureml://datastores/${DSCOP_DATASTORE_NAME}/paths/xbt/xbt_1968.csv"
export CLIMATE_ZONES_AML_URI="azureml://datastores/${DSCOP_DATASTORE_NAME}/paths/climate_zones/climate_zones_1p0.csv"

python create_workspace_spec.py dataset --name xbt_sample --path $XBT_AML_URI --type uri_file --output dataset_xbt_spec.yaml
az ml data create --file dataset_xbt_spec.yaml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

python create_workspace_spec.py dataset --name climate_zones_1_0 --path $CLIMATE_ZONES_AML_URI --type uri_file --output dataset_cz_spec.yaml
az ml data create --file dataset_cz_spec.yaml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME


#=========================================
# Create compute instances and clusters
export COMPUTE_SIZE=Standard_DS3_v2

# instance
# copy the script to user space

export FILE_STORE=code-391ff5ac-6576-460f-ba4d-7e03433c68b6
azcopy copy aml_ci_setup.sh  "https://${STORAGE_ACCOUNT}.file.core.windows.net/${FILE_STORE}/Users/${USER_NAME}/aml_ci_setup.sh"

az ml compute create --name dscopcitest01 --size $COMPUTE_SIZE --type ComputeInstance --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
#todo: specify the script to run on creation to set up the copmpute instance for the tutorial, need to create a python script to populate a compute spec yaml file which specifies a creation script, which must first be uploaded

#cluster
az ml compute create --name dscopclustertest01 --size $COMPUTE_SIZE --min-instances 0 --max-instances 1 --type AmlCompute --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME


#todo: run command to create file from template
az ml environment create -f environments/azure_templates/envs/keras/env_keras_aml.yaml -w mlw-dscoptrainingenv-uksouth-01 -g rg-dscop-dscoptrainingenv

# SECTION - set up datasets


#todo wget weatherbench 5 degree snapshot data

# upload to azure blob storage


az storage account list
export STORAGE_ACCOUNT=$(python get_storage_account.py $RESOURCE_GROUP --first-only)
export CONTAINER_NAME=azureml
export CONTAINER_URL= https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER_NAME}

export XBT_DATA_PATH=/data/users/dscop/ml_tutorial/xbt/xbt_1968.csv
azcopy copy ${XBT_DATA_PATH} ${CONTAINER_URL}/xbt/xbt_1968.csv


if [ "${PLATFORM}" = "metoffice" ]; then
  export XBT_DATA_PATH=/data/users/dscop/ml_tutorial/xbt/xbt_1968.csv
  export CLIMATE_ZONES_DATA_PATH=/data/users/dscop/ml_tutorial/climate_zones/ml_ready/climate_zones_1p0.csv
else
  export DATA_PATH=./data
  export XBT_DATA_PATH="${DATA_PATH}/xbt/xbt_1968.csv"
  export CLIMATE_ZONES_DATA_PATH="${DATA_PATH}/climate_zones/ml_ready/climate_zones_1p0.csv"

  mkdir -p "$(dirname "${XBT_DATA_PATH}")" "$(dirname "${CLIMATE_ZONES_DATA_PATH}")"
  wget -O "${XBT_DATA_PATH}" https://zenodo.org/records/7390654/files/xbt_1968.csv
  wget -O "${CLIMATE_ZONES_DATA_PATH}" https://zenodo.org/records/21773439/files/climate_zones_1p0.csv
fi

azcopy copy "${XBT_DATA_PATH}" "${CONTAINER_URL}/xbt/xbt_1968.csv"
azcopy copy "${CLIMATE_ZONES_DATA_PATH}" "${CONTAINER_URL}/climate_zones/climate_zones_1p0.csv"

#download the data
wget https://zenodo.org/records/7390654/files/xbt_1968.csv
wget https://zenodo.org/records/21773439/files/climate_zones_1p0.csv

export CLIMATE_ZONES_DATA_PATH=/data/users/dscop/ml_tutorial/climate_zones/ml_ready/climate_zones_1p0.csv
azcopy copy ${CLIMATE_ZONES_DATA_PATH} https://${STORAGE_ACCOUNT}.blob.core.windows.net/${CONTAINER_NAME}/climate_zones/climate_zones_1p0.csv

# todo: copy weatherbench data


# create azure datastores
#todo: create datastores for xbt and climate zones and weatherbench
#TODO: create datastore yaml file
az ml datastore create --name <datastore_name> --type azureblob --account-name <storage_account_name> --container-name <container_name> --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME

# create azure datasets
#TODO: create dataset yaml file
az ml dataset create --name <dataset_name> --type tabular --path <datastore_name>/<path_to_data> --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME