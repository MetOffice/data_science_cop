import os

import azure.ai.ml
import azure.identity
import azureml.fsspec

import xarray


credential=azure.identity.DefaultAzureCredential()


# ml_client = azure.ai.ml.MLClient.from_config(credential=credential)
ml_client = azure.ai.ml.MLClient(
    credential=credential,
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZUREML_WORKSPACE_NAME"],
)

print(ml_client)

datastore_name='workspaceblobstore'
datastore_uri = f'azureml://subscriptions/{ml_client.subscription_id}/resourcegroups/{ml_client.resource_group_name}/workspaces/{ml_client.workspace_name}/datastores/{datastore_name}/paths'
wb_uri = datastore_uri + '/weatherbench/5.625deg/temperature/temperature_1980_5.625deg.nc'

# Connect using the AzureML datastore URI
fs = azureml.fsspec.AzureMachineLearningFileSystem(wb_uri)

with fs.open(wb_uri, mode="rb") as f:
    ds = xarray.open_dataset(f, engine="h5netcdf")
print(ds)

