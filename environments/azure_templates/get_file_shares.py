#!/usr/bin/env python3
"""List Azure file share names from `az storage share list`.

The script calls the Azure CLI for a specific storage account, parses the JSON
output, and prints matching file share names one per line to standard output.
Use `--prefix` to return only shares whose names start with the provided text.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any, Optional


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="List Azure file shares for a storage account."
    )
    parser.add_argument(
        "storage_account",
        help="Azure Storage account name to list file shares from.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help="Optional prefix filter for file share names.",
    )
    return parser.parse_args(argv)


def fetch_file_shares(storage_account: str) -> list[dict[str, Any]]:
    """Fetch Azure file shares for a storage account via Azure CLI.

    Args:
        storage_account: Storage account name to query.

    Returns:
        A list of file share objects as dictionaries.

    Raises:
        RuntimeError: If the Azure CLI command fails or returns invalid JSON.
    """
    command = [
        "az",
        "storage",
        "share",
        "list",
        "--account-name",
        storage_account,
        "--output",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            "Failed to run Azure CLI command 'az storage share list': "
            f"{stderr or f'exit code {result.returncode}'}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Azure CLI returned invalid JSON output.") from exc

    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Azure CLI output: expected a JSON list.")

    return payload


def extract_share_names(file_shares: list[dict[str, Any]]) -> list[str]:
    """Extract file share names from Azure CLI response objects.

    Args:
        file_shares: List of file share dictionaries.

    Returns:
        A list containing non-empty file share names.
    """
    names: list[str] = []
    for file_share in file_shares:
        name = file_share.get("name")
        if isinstance(name, str) and name:
            names.append(name)
    return names


def filter_names_by_prefix(names: list[str], prefix: Optional[str]) -> list[str]:
    """Filter file share names by a prefix.

    Args:
        names: List of file share names.
        prefix: Prefix to filter on. If None or empty, no filtering is applied.

    Returns:
        Filtered list of file share names.
    """
    if not prefix:
        return names
    return [name for name in names if name.startswith(prefix)]


def main(argv: Optional[list[str]] = None) -> int:
    """Run the command-line entry point.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)

    try:
        file_shares = fetch_file_shares(args.storage_account)
        names = extract_share_names(file_shares)
        filtered_names = filter_names_by_prefix(names, args.prefix)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    for name in filtered_names:
        print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

