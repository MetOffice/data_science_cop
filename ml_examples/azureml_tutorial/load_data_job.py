#!/usr/bin/env python
"""Train a climate-zone classifier with PyTorch and track runs in MLflow.

This script refactors the original notebook export into a standard Python CLI.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot

import mlflow

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import azureml.core
import azureml.fsspec




def get_azureml_data_path():
    """Resolve root data directory from config platform mapping."""
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())

    aml_workspace = azureml.core.Workspace(ml_client.subscription_id, ml_client.resource_group_name,
                                           ml_client.workspace_name)

    datastore_uri = f'azureml://subscriptions/{ml_client.subscription_id}/resourcegroups/{ml_client.resource_group_name}/workspaces/{ml_client.workspace_name}/datastores/{datastore_name}/paths'
    climate_zones_uri = datastore_uri + '/climate_zones/ml_ready'

    return climate_zones_uri

def get_mlflow():
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    mlflow_server_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
    print(f'connecting to mlflow server {mlflow_server_uri}')
    mlflow.set_tracking_uri(mlflow_server_uri)
    return mlflow_server_uri

def load_data():
    """Load climate zone data from AzureML datastore."""
    climate_zones_uri = get_azureml_data_path()
    print(f'loading data from climate_zones uri {climate_zones_uri}')
    cz_df = pd.read_csv(climate_zones_uri)
    return cz_df

def plot_data(cz_df):
    exp_name = azureml.core.Run.get_context().experiment.name
    print(f'using experiment {exp_name}')
    exp1 = mlflow.get_experiment_by_name(exp_name)
    with mlflow.start_run(experiment_id=exp1.experiment_id) as current_run:
        class_counts = cz_df['climate_group'].value_counts()
        current_run.log_dict(class_counts.to_dict())
        fig1 = matplotlib.pyplot.figure()
        ax1 = fig1.add_subplots(1,1,1)
        class_counts.plot.bar(ax=ax1)
        current_run.log_figure(fig1, 'class_counts.png')

def main() -> None:
    """Parse args and run training."""
    mlflow_server_uri = get_mlflow()
    cz_df = load_data()
    plot_data(cz_df)

if __name__ == "__main__":
    main()
