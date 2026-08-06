#!/usr/bin/env python3
"""List Azure resource group names from `az group list`.

The script calls the Azure CLI, parses the JSON output, and prints the
resource group names one per line to standard output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any


def fetch_resource_groups() -> list[dict[str, Any]]:
    """Fetch all Azure resource groups via the Azure CLI.

    Returns:
        A list of resource group objects as dictionaries.

    Raises:
        RuntimeError: If the Azure CLI command fails or returns invalid JSON.
    """
    command = ["az", "group", "list", "--output", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            "Failed to run Azure CLI command 'az group list': "
            f"{stderr or f'exit code {result.returncode}'}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Azure CLI returned invalid JSON output.") from exc

    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Azure CLI output: expected a JSON list.")

    return payload


def extract_resource_group_names(resource_groups: list[dict[str, Any]]) -> list[str]:
    """Extract resource group names from Azure CLI response objects.

    Args:
        resource_groups: List of resource group dictionaries.

    Returns:
        A list containing the non-empty resource group names.
    """
    names: list[str] = []
    for resource_group in resource_groups:
        name = resource_group.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def main() -> int:
    """Run the command-line entry point.

    Returns:
        Process exit code.
    """
    try:
        resource_groups = fetch_resource_groups()
        names = extract_resource_group_names(resource_groups)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for name in names:
        print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
