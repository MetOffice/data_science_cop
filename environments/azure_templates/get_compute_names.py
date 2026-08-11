#!/usr/bin/env python3
"""List Azure ML compute names from `az ml compute list`.

The script calls the Azure CLI for a specific Azure ML workspace, parses the
JSON output, and prints matching compute names one per line to standard output.
Use `--compute-type` to return only computes of a given type.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Optional


COMPUTE_TYPE_CHOICES = ("AmlCompute", "ComputeInstance")

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="List Azure ML compute names for a workspace."
    )
    parser.add_argument(
        "resource_group",
        help="Azure resource group containing the Azure ML workspace.",
    )
    parser.add_argument(
        "workspace_name",
        help="Azure ML workspace name to list computes from.",
    )
    parser.add_argument(
        "-t",
        "--compute-type",
        choices=COMPUTE_TYPE_CHOICES,
        help="Optional compute type filter.",
    )
    return parser.parse_args(argv)


def fetch_computes(resource_group: str, workspace_name: str) -> list[dict[str, Any]]:
    """Fetch Azure ML computes for a workspace via Azure CLI.

    Args:
        resource_group: Azure resource group name.
        workspace_name: Azure ML workspace name.

    Returns:
        A list of compute objects as dictionaries.

    Raises:
        RuntimeError: If the Azure CLI command fails or returns invalid JSON.
    """
    command = [
        "az",
        "ml",
        "compute",
        "list",
        "--resource-group",
        resource_group,
        "--workspace-name",
        workspace_name,
        "--output",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            "Failed to run Azure CLI command 'az ml compute list': "
            f"{stderr or f'exit code {result.returncode}'}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Azure CLI returned invalid JSON output.") from exc

    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Azure CLI output: expected a JSON list.")

    return payload


def extract_compute_names(computes: list[dict[str, Any]]) -> list[str]:
    """Extract compute names from Azure CLI response objects.

    Args:
        computes: List of compute dictionaries.

    Returns:
        A list containing non-empty compute names.
    """
    names: list[str] = []
    for compute in computes:
        name = compute.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def filter_computes_by_type(
    computes: list[dict[str, Any]], compute_type: Optional[str]
) -> list[dict[str, Any]]:
    """Filter compute objects by compute type.

    Args:
        computes: List of compute dictionaries.
        compute_type: Compute type filter. If None, no filtering is applied.

    Returns:
        Filtered list of compute dictionaries.
    """
    if not compute_type:
        return computes

    return [
        compute
        for compute in computes
        if isinstance(compute.get("type"), str) and compute.get("type").lower() == compute_type.lower()
    ]


def main(argv: Optional[list[str]] = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    try:
        computes = fetch_computes(args.resource_group, args.workspace_name)
        filtered_computes = filter_computes_by_type(computes, args.compute_type)
        names = extract_compute_names(filtered_computes)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for name in names:
        print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

