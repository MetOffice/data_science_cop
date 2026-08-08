#!/usr/bin/env bash
# This script is to be run from any machine that
# has the Azure CLI installed and configured to create an AzureML workspace
# and associated resources for the DS CoP AzureML tutorials.

# Login to AzureML CLI - If your terminal is not currently running in a logged-in Azure CLI session, run the following command to log in:
# az login

# the user name for the account must be specified by the user through a command line argument, otherwise the rest of the script will not work.
# Check if the first argument is empty
if [ -z "$1" ]; then
    echo "Error: You must supply the username for the azure account in the first command line argument." >&2
    exit 1
else
    export USER_NAME=$1
fi

#use this if on Met Office IT
export PLATFORM=metoffice

# use for other
# export PLATFORM=local


export WORKSPACE_NAME=dscoptest1
export RESOURCE_GROUP=$(python get_resource_groups.py)

#===============================
# Create workspace

python create_workspace_spec.py workspace --name $WORKSPACE_NAME --resource-group $RESOURCE_GROUP --description "Workspace for DS CoP AzureML tutorials." --output workspace_spec.yaml
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

#export FILE_STORE=code-391ff5ac-6576-460f-ba4d-7e03433c68b6
#azcopy copy aml_ci_setup.sh  "https://${STORAGE_ACCOUNT}.file.core.windows.net/${FILE_STORE}/Users/${USER_NAME}/aml_ci_setup.sh"

az ml compute create --name dscopcitest01 --size $COMPUTE_SIZE --type ComputeInstance --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME
#todo: specify the script to run on creation to set up the copmpute instance for the tutorial, need to create a python script to populate a compute spec yaml file which specifies a creation script, which must first be uploaded

#cluster
#az ml compute create --name dscopclustertest01 --size $COMPUTE_SIZE --min-instances 0 --max-instances 1 --type AmlCompute --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME


#=========================================
# Create compute instances and clusters
#python create_workspace_spec.py environment  --conda-file ../requirements_pytorch.yml --name dscop_pytorch --output env_dscop_pytorch_spec.yaml --description "Environment for DS CoP AzureML tutorials with PyTorch and other dependencies." --image  "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
#
#az ml environment create --file env_dscop_pytorch_spec.yaml --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME


