import os

import azure.ai.ml
import azure.identity

import pandas

print('getting credentials')
credential=azure.identity.DefaultAzureCredential()


# ml_client = azure.ai.ml.MLClient.from_config(credential=credential)
ml_client = azure.ai.ml.MLClient(
    credential=credential,
    subscription_id=os.environ["AZURE_SUBSCRIPTION_ID"],
    resource_group_name=os.environ["AZURE_RESOURCE_GROUP"],
    workspace_name=os.environ["AZUREML_WORKSPACE_NAME"],
)

print('retieved client credentials')
print(ml_client)

datastore_name = 'workspaceblobstore'
datastore_uri = f'azureml://subscriptions/{ml_client.subscription_id}/resourcegroups/{ml_client.resource_group_name}/workspaces/{ml_client.workspace_name}/datastores/{datastore_name}/paths'
print(f'datastore {datastore_uri}')
climate_zones_uri = datastore_uri + '/climate_zones'
mlready_data_uri = climate_zones_uri + '/ml_ready'
cz_1p0_file = mlready_data_uri + '/climate_zones_1p0.csv'
print(f'file uri {cz_1p0_file}')

print('reading csv')
df1 = pandas.read_csv(cz_1p0_file)
print(df1)

