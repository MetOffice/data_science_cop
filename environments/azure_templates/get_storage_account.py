#!/usr/bin/env python
"""List Azure Storage Account names for a given resource group.

This script calls `az storage account list --output json`, filters accounts
by resource group, and prints matching storage account names to stdout.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "List Azure storage account names for a specific resource group."
        )
    )
    parser.add_argument(
        "resource_group",
        help="Azure resource group name to filter storage accounts by.",
    )
    parser.add_argument(
        "-f",
        "--first-only",
        action="store_true",
        help="Print only the first matching storage account name.",
    )
    return parser.parse_args(argv)


def fetch_storage_accounts() -> list[dict[str, Any]]:
    """Fetch all Azure storage accounts via Azure CLI.

    Returns:
        A list of storage account objects as dictionaries.

    Raises:
        RuntimeError: If the Azure CLI command fails or returns invalid JSON.
    """
    command = ["az", "storage", "account", "list", "--output", "json"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            "Failed to run Azure CLI command 'az storage account list': "
            f"{stderr or f'exit code {result.returncode}'}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Azure CLI returned invalid JSON output.") from exc

    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Azure CLI output: expected a JSON list.")

    return payload


def filter_accounts_by_resource_group(
    accounts: list[dict[str, Any]], resource_group: str
) -> list[dict[str, Any]]:
    """Filter storage account records by resource group name.

    Args:
        accounts: List of storage account dictionaries.
        resource_group: Resource group name to filter on.

    Returns:
        Filtered list of storage account dictionaries.
    """
    target = resource_group.casefold()
    return [
        account
        for account in accounts
        if str(account.get("resourceGroup", "")).casefold() == target
    ]


def extract_account_names(accounts: list[dict[str, Any]]) -> list[str]:
    """Extract storage account names from account dictionaries.

    Args:
        accounts: List of storage account dictionaries.

    Returns:
        List of account names.
    """
    names: list[str] = []
    for account in accounts:
        name = account.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    try:
        accounts = fetch_storage_accounts()
        filtered = filter_accounts_by_resource_group(accounts, args.resource_group)
        names = extract_account_names(filtered)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.first_only:
        if names:
            print(names[0])
        return 0

    for name in names:
        print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

