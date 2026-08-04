#!/usr/bin/env bash
# This script is to be run from any machine that
# has the Azure CLI installed and configured to create an AzureML workspace
# and associated resources for the DS CoP AzureML tutorials.

# Login to AzureML CLI - If your terminal is not currently running in a logged-in Azure CLI session, run the following command to log in:
# az login

export WORKSPACE_NAME=1-d69dad24-playground-sandbox
export RESOURCE_GROUP=dscoptest1


#todo: run command to create file from template
az ml workspace create --file workspace.yml --resource-group $RESOURCE_GROUP

# create workspace
# todo  - create workspace yaml file
# todo create workspace with command az ml workspace create --file workspace.yml --resource-group $RESOURCE_GROUP

# instance
az ml compute create --name dscoptest1 --size Standard_DS3_v2 --type ComputeInstance --resource-group 1-d69dad24-playground-sandbox --workspace-name
#todo: specify the script to run on creation to set up the copmpute instance for the tutorial

#cluster
az ml compute create --name dscopclustertest01 --size Standard_DS3_v2 --min-instances 0 --max-instances 1 --type AmlCompute --resource-group 1-d69dad24-playground-sandbox --workspace-name dscoptest1


#todo: run command to create file from template
az ml environment create -f environments/azure_templates/envs/keras/env_keras_aml.yaml -w mlw-dscoptrainingenv-uksouth-01 -g rg-dscop-dscoptrainingenv

