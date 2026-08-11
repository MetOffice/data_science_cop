"""
Generate Azure ML YAML specifications from the command line.

This module supports creating YAML specs for:
- Azure ML workspace
- Azure ML environment (from a conda file)
- Azure ML Azure Blob datastore
- Azure ML data asset (dataset)
- Azure ML command job

References:
- Workspace schema:
  https://learn.microsoft.com/en-gb/azure/machine-learning/reference-yaml-workspace?view=azureml-api-2
- Environment schema:
  https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-environment?view=azureml-api-2
- Datastore (blob) schema:
  https://learn.microsoft.com/en-gb/azure/machine-learning/reference-yaml-datastore-blob?view=azureml-api-2
- Data asset schema:
  https://azuremlschemas.azureedge.net/latest/data.schema.json
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


WORKSPACE_SCHEMA = "https://azuremlschemas.azureedge.net/latest/workspace.schema.json"
ENVIRONMENT_SCHEMA = "https://azuremlschemas.azureedge.net/latest/environment.schema.json"
DATASTORE_BLOB_SCHEMA = "https://azuremlschemas.azureedge.net/latest/datastoreBlob.schema.json"
DATA_SCHEMA = "https://azuremlschemas.azureedge.net/latest/data.schema.json"
COMMAND_JOB_SCHEMA = "https://azuremlschemas.azureedge.net/latest/commandJob.schema.json"


def write_yaml_file(spec: Dict[str, Any], output_path: str) -> bool:
    """Write a spec dictionary to disk as YAML.

    Args:
        spec: YAML content as a dictionary.
        output_path: Destination file path.

    Returns:
        True if write succeeds, otherwise False.
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with output_file.open("w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(spec, yaml_file, default_flow_style=False, sort_keys=False)

        print(f"Successfully created specification file: {output_file}")
        return True

    except IOError as err:
        print(f"Error writing to file {output_path}: {err}")
        return False


def create_workspace_yaml(name: str, resource_group: str, output_path: str, description: str) -> bool:
    """Create an Azure ML workspace YAML specification file."""
    workspace_spec = {
        "$schema": WORKSPACE_SCHEMA,
        "name": name,
        "resource_group": resource_group,
        "description": description,
    }
    return write_yaml_file(workspace_spec, output_path)


def create_environment_yaml(
    name: str,
    conda_file: str,
    output_path: str,
    version: Optional[str],
    image: Optional[str],
    description: Optional[str],
) -> bool:
    """Create an Azure ML environment YAML specification file from a conda file."""
    conda_path = Path(conda_file)
    if not conda_path.exists():
        print(f"Conda file does not exist: {conda_path}")
        return False

    environment_spec: Dict[str, Any] = {
        "$schema": ENVIRONMENT_SCHEMA,
        "name": name,
        "conda_file": str(conda_path),
    }

    # Add optional fields only when supplied so output stays concise.
    if version:
        environment_spec["version"] = version
    if image:
        environment_spec["image"] = image
    if description:
        environment_spec["description"] = description

    return write_yaml_file(environment_spec, output_path)


def create_datastore_yaml(
    name: str,
    account_name: str,
    container_name: str,
    output_path: str,
    protocol: Optional[str],
    endpoint: Optional[str],
    description: Optional[str],
    account_key: Optional[str],
    sas_token: Optional[str],
) -> bool:
    """Create an Azure ML Azure Blob datastore YAML specification file."""
    datastore_spec: Dict[str, Any] = {
        "$schema": DATASTORE_BLOB_SCHEMA,
        "name": name,
        "type": "azure_blob",
        "account_name": account_name,
        "container_name": container_name,
    }

    if protocol:
        datastore_spec["protocol"] = protocol
    if endpoint:
        datastore_spec["endpoint"] = endpoint
    if description:
        datastore_spec["description"] = description

    # Credentials are optional; include whichever the user provided.
    credentials: Dict[str, str] = {}
    if account_key:
        credentials["account_key"] = account_key
    if sas_token:
        credentials["sas_token"] = sas_token
    if credentials:
        datastore_spec["credentials"] = credentials

    return write_yaml_file(datastore_spec, output_path)


def create_dataset_yaml(
    name: str,
    path: str,
    output_path: str,
    data_type: str,
    version: Optional[str],
    description: Optional[str],
) -> bool:
    """Create an Azure ML data asset (dataset) YAML specification file."""
    dataset_spec: Dict[str, Any] = {
        "$schema": DATA_SCHEMA,
        "name": name,
        "type": data_type,
        "path": path,
    }

    if version:
        dataset_spec["version"] = version
    if description:
        dataset_spec["description"] = description

    return write_yaml_file(dataset_spec, output_path)


def create_job_yaml(
    code: str,
    command: str,
    compute_name: str,
    output_path: str,
    environment: Optional[str],
    experiment_name: Optional[str],
    job_name: Optional[str],
) -> bool:
    """Create an Azure ML command job YAML specification file."""
    code_path = Path(code)
    if not code_path.exists() or not code_path.is_dir():
        print(f"Code directory does not exist or is not a directory: {code_path}")
        return False

    # Azure ML command jobs expect compute in the form `azureml:<compute_name>`.
    compute_target = (
        compute_name
        if compute_name.startswith("azureml:")
        else f"azureml:{compute_name}"
    )

    job_spec: Dict[str, Any] = {
        "$schema": COMMAND_JOB_SCHEMA,
        "type": "command",
        "code": str(code_path),
        "command": command,
        "compute": compute_target,
    }

    if environment:
        job_spec["environment"] = environment
    if experiment_name:
        job_spec["experiment_name"] = experiment_name
    if job_name:
        job_spec["name"] = job_name

    return write_yaml_file(job_spec, output_path)


def add_workspace_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the workspace subcommand and its arguments."""
    workspace_parser = subparsers.add_parser(
        "workspace", help="Create an Azure ML workspace YAML specification"
    )
    workspace_parser.add_argument("--name", type=str, required=True, help="Workspace name")
    workspace_parser.add_argument(
        "--resource-group",
        dest="resource_group",
        type=str,
        required=True,
        help="Resource group name",
    )
    workspace_parser.add_argument(
        "--description",
        type=str,
        default="Workspace for Met Office DS CoP learning and development purposes.",
        help="Optional workspace description",
    )
    workspace_parser.add_argument(
        "--output",
        type=str,
        default="workspace_spec.yaml",
        help="Output YAML path",
    )
    workspace_parser.set_defaults(func=handle_workspace_subcommand)


def add_environment_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the environment subcommand and its arguments."""
    environment_parser = subparsers.add_parser(
        "environment",
        help="Create an Azure ML environment YAML specification from a conda file",
    )
    environment_parser.add_argument("--name", type=str, required=True, help="Environment name")
    environment_parser.add_argument(
        "--conda-file",
        dest="conda_file",
        type=str,
        required=True,
        help="Path to conda YAML file",
    )
    environment_parser.add_argument("--version", type=str, help="Optional environment version")
    environment_parser.add_argument(
        "--image",
        type=str,
        help="Optional base Docker image (for example mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04)",
    )
    environment_parser.add_argument("--description", type=str, help="Optional environment description")
    environment_parser.add_argument(
        "--output",
        type=str,
        default="environment_spec.yaml",
        help="Output YAML path",
    )
    environment_parser.set_defaults(func=handle_environment_subcommand)


def add_datastore_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the datastore subcommand and its arguments."""
    datastore_parser = subparsers.add_parser(
        "datastore", help="Create an Azure ML Azure Blob datastore YAML specification"
    )
    datastore_parser.add_argument("--name", type=str, required=True, help="Datastore name")
    datastore_parser.add_argument(
        "--account-name",
        dest="account_name",
        type=str,
        required=True,
        help="Azure Storage account name",
    )
    datastore_parser.add_argument(
        "--container-name",
        dest="container_name",
        type=str,
        required=True,
        help="Blob container name",
    )
    datastore_parser.add_argument(
        "--protocol",
        choices=["https", "wasbs"],
        help="Optional storage protocol",
    )
    datastore_parser.add_argument(
        "--endpoint",
        type=str,
        help="Optional storage endpoint suffix (for example core.windows.net)",
    )
    datastore_parser.add_argument("--description", type=str, help="Optional datastore description")

    credential_group = datastore_parser.add_mutually_exclusive_group()
    credential_group.add_argument(
        "--account-key",
        dest="account_key",
        type=str,
        help="Optional account key credential",
    )
    credential_group.add_argument(
        "--sas-token",
        dest="sas_token",
        type=str,
        help="Optional SAS token credential",
    )

    datastore_parser.add_argument(
        "--output",
        type=str,
        default="datastore_spec.yaml",
        help="Output YAML path",
    )
    datastore_parser.set_defaults(func=handle_datastore_subcommand)


def add_dataset_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the dataset subcommand and its arguments."""
    dataset_parser = subparsers.add_parser(
        "dataset", help="Create an Azure ML data asset (dataset) YAML specification"
    )
    dataset_parser.add_argument("--name", type=str, required=True, help="Dataset/data asset name")
    dataset_parser.add_argument(
        "--path",
        type=str,
        required=True,
        help=(
            "Path to data source (for example local path, URI, or "
            "azureml://datastores/<name>/paths/<path>)"
        ),
    )
    dataset_parser.add_argument(
        "--type",
        dest="data_type",
        type=str,
        required=True,
        choices=["uri_file", "uri_folder", "mltable"],
        help="Data asset type",
    )
    dataset_parser.add_argument("--version", type=str, help="Optional data asset version")
    dataset_parser.add_argument("--description", type=str, help="Optional data asset description")
    dataset_parser.add_argument(
        "--output",
        type=str,
        default="dataset_spec.yaml",
        help="Output YAML path",
    )
    dataset_parser.set_defaults(func=handle_dataset_subcommand)


def add_job_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """Register the command job subcommand and its arguments."""
    job_parser = subparsers.add_parser(
        "job", help="Create an Azure ML command job YAML specification"
    )
    job_parser.add_argument(
        "--code",
        type=str,
        required=True,
        help="Path to local code directory to upload",
    )
    job_parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="Command string to run",
    )
    job_parser.add_argument(
        "--compute-name",
        dest="compute_name",
        type=str,
        required=True,
        help="Compute target name (compute instance or cluster)",
    )
    job_parser.add_argument(
        "--environment",
        type=str,
        help="Optional environment reference (for example azureml:my-env@latest)",
    )
    job_parser.add_argument(
        "--experiment-name",
        dest="experiment_name",
        type=str,
        help="Optional experiment name",
    )
    job_parser.add_argument(
        "--job-name",
        dest="job_name",
        type=str,
        help="Optional job name",
    )
    job_parser.add_argument(
        "--output",
        type=str,
        default="job_spec.yaml",
        help="Output YAML path",
    )
    job_parser.set_defaults(func=handle_job_subcommand)


def get_cmd_args() -> argparse.Namespace:
    """Parse and return command line arguments for all supported subcommands."""
    parser = argparse.ArgumentParser(
        description="Create Azure ML YAML specification files"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_workspace_subcommand(subparsers)
    add_environment_subcommand(subparsers)
    add_datastore_subcommand(subparsers)
    add_dataset_subcommand(subparsers)
    add_job_subcommand(subparsers)

    return parser.parse_args()


def handle_workspace_subcommand(args: argparse.Namespace) -> bool:
    """Handle `workspace` subcommand execution."""
    return create_workspace_yaml(
        name=args.name,
        resource_group=args.resource_group,
        output_path=args.output,
        description=args.description,
    )


def handle_environment_subcommand(args: argparse.Namespace) -> bool:
    """Handle `environment` subcommand execution."""
    return create_environment_yaml(
        name=args.name,
        conda_file=args.conda_file,
        output_path=args.output,
        version=args.version,
        image=args.image,
        description=args.description,
    )


def handle_datastore_subcommand(args: argparse.Namespace) -> bool:
    """Handle `datastore` subcommand execution."""
    return create_datastore_yaml(
        name=args.name,
        account_name=args.account_name,
        container_name=args.container_name,
        output_path=args.output,
        protocol=args.protocol,
        endpoint=args.endpoint,
        description=args.description,
        account_key=args.account_key,
        sas_token=args.sas_token,
    )


def handle_dataset_subcommand(args: argparse.Namespace) -> bool:
    """Handle `dataset` subcommand execution."""
    return create_dataset_yaml(
        name=args.name,
        path=args.path,
        output_path=args.output,
        data_type=args.data_type,
        version=args.version,
        description=args.description,
    )


def handle_job_subcommand(args: argparse.Namespace) -> bool:
    """Handle `job` subcommand execution."""
    return create_job_yaml(
        code=args.code,
        command=args.command,
        compute_name=args.compute_name,
        output_path=args.output,
        environment=args.environment,
        experiment_name=args.experiment_name,
        job_name=args.job_name,
    )


def main() -> None:
    """Main entry point for Azure ML specification creation."""
    args = get_cmd_args()
    success = args.func(args)
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()