#!/usr/bin/env bash
#
# Bash wrapper for test_climatezones_data_exploration.py.
#
# Loads the community environment module, then invokes the Python test script,
# propagating its exit code unchanged. Contains no test logic and no
# Cylc-specific behaviour; runnable manually now, and later from Cylc.
#
# Usage:
#   ./run_test_climatezones_data_exploration.sh [--module <module>] [--troubleshooting]
#
# Default module: scitools/community/ml
# --troubleshooting: activates the Python script's artefact-retention mode (off by default).

set -eu

MODULE="scitools/community/ml"
TROUBLESHOOTING=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --module)
            MODULE="$2"
            shift 2
            ;;
        --troubleshooting)
            TROUBLESHOOTING=1
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

module load "$MODULE" || exit 1

echo "Module: $MODULE"
if [[ "$TROUBLESHOOTING" -eq 1 ]]; then
    echo "Troubleshooting: ON"
else
    echo "Troubleshooting: OFF"
fi
echo

cd "$SCRIPT_DIR"
ARGS=(--module "$MODULE")
if [[ "$TROUBLESHOOTING" -eq 1 ]]; then
    ARGS+=(--troubleshooting)
fi
python test_climatezones_data_exploration.py "${ARGS[@]}"
