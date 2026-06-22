#!/usr/bin/env bash
set -euo pipefail

SUBSCRIPTION_ID="<your-subscription-id>"
RESOURCE_GROUP="rg-azureml-arm-demo"
LOCATION="uksouth"

WORKSPACE_NAME="mlw-arm-demo"
STORAGE_ACCOUNT_NAME="stmlarmdemo$RANDOM"
KEYVAULT_NAME="kv-ml-arm-demo-$RANDOM"
ACR_NAME="acrmlarmdemo$RANDOM"
APPINSIGHTS_NAME="appi-ml-arm-demo"

COMPUTE_OWNER_OBJECT_ID="<your-entra-user-object-id>"

DEPLOYMENT_STORAGE_CONTAINER="aml-env-build-contexts"

az account set --subscription "$SUBSCRIPTION_ID"

az group create \
  --name "$RESOURCE_GROUP" \
  --location "$LOCATION"

az storage account create \
  --name "$STORAGE_ACCOUNT_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --allow-shared-key-access true

STORAGE_KEY=$(az storage account keys list \
  --resource-group "$RESOURCE_GROUP" \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --query "[0].value" \
  --output tsv)

az storage container create \
  --name "$DEPLOYMENT_STORAGE_CONTAINER" \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --account-key "$STORAGE_KEY"

tar -czf keras-context.tar.gz -C envs/keras .
tar -czf lightning-context.tar.gz -C envs/lightning .

az storage blob upload \
  --container-name "$DEPLOYMENT_STORAGE_CONTAINER" \
  --name keras-context.tar.gz \
  --file keras-context.tar.gz \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --account-key "$STORAGE_KEY" \
  --overwrite true

az storage blob upload \
  --container-name "$DEPLOYMENT_STORAGE_CONTAINER" \
  --name lightning-context.tar.gz \
  --file lightning-context.tar.gz \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --account-key "$STORAGE_KEY" \
  --overwrite true

EXPIRY=$(date -u -v+2d '+%Y-%m-%dT%H:%MZ' 2>/dev/null || date -u -d '+2 days' '+%Y-%m-%dT%H:%MZ')

KERAS_SAS=$(az storage blob generate-sas \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --account-key "$STORAGE_KEY" \
  --container-name "$DEPLOYMENT_STORAGE_CONTAINER" \
  --name keras-context.tar.gz \
  --permissions r \
  --expiry "$EXPIRY" \
  --output tsv)

LIGHTNING_SAS=$(az storage blob generate-sas \
  --account-name "$STORAGE_ACCOUNT_NAME" \
  --account-key "$STORAGE_KEY" \
  --container-name "$DEPLOYMENT_STORAGE_CONTAINER" \
  --name lightning-context.tar.gz \
  --permissions r \
  --expiry "$EXPIRY" \
  --output tsv)

KERAS_CONTEXT_URI="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${DEPLOYMENT_STORAGE_CONTAINER}/keras-context.tar.gz?${KERAS_SAS}"
LIGHTNING_CONTEXT_URI="https://${STORAGE_ACCOUNT_NAME}.blob.core.windows.net/${DEPLOYMENT_STORAGE_CONTAINER}/lightning-context.tar.gz?${LIGHTNING_SAS}"

az deployment group create \
  --name "azureml-arm-deployment" \
  --resource-group "$RESOURCE_GROUP" \
  --template-file azuredeploy.json \
  --parameters \
    location="$LOCATION" \
    workspaceName="$WORKSPACE_NAME" \
    storageAccountName="$STORAGE_ACCOUNT_NAME" \
    keyVaultName="$KEYVAULT_NAME" \
    acrName="$ACR_NAME" \
    appInsightsName="$APPINSIGHTS_NAME" \
    computeInstanceOwnerObjectId="$COMPUTE_OWNER_OBJECT_ID" \
    kerasBuildContextUri="$KERAS_CONTEXT_URI" \
    lightningBuildContextUri="$LIGHTNING_CONTEXT_URI"

az extension add --name ml --upgrade

az ml workspace show \
  --name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --output table

az ml environment list \
  --workspace-name "$WORKSPACE_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --output table