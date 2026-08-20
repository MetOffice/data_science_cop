#!/usr/bin/env bash
#
# Bash wrapper for test_climatezones_data_exploration.py.
#
# Loads the community environment module, then invokes the Python test script,
# propagating its exit code unchanged. Contains no test logic and no
# Cylc-specific behaviour; runnable manually now, and later from Cylc.
#
# Usage:
#   ./run_test_climatezones_data_exploration.sh [--module <module>] [--retention]
#
# Default module: scitools/community/ml
# --retention: activates the Python script's artefact-retention mode (off by default).
set -eu

MODULE="scitools/community/ml"
RETENTION=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --module)
            MODULE="$2"
            shift 2
            ;;
        --retention)
            RETENTION=1
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

ENVIRONMENT_INVENTORY=$(
    find "$SSS_ENV_DIR/conda-meta" \
        -maxdepth 1 \
        -type f \
        -name '*.json' \
        -printf '%f\n' |
    sort
)

ENVIRONMENT_HASH=$(
    printf "%s" "$ENVIRONMENT_INVENTORY" |
    sha256sum |
    awk '{print $1}'
)

echo "Module: $MODULE"
if [[ "$RETENTION" -eq 1 ]]; then
    echo "Retention: ON"
else
    echo "Retention: OFF"
fi
echo "Environment Hash: $ENVIRONMENT_HASH"
echo

cd "$SCRIPT_DIR"
ARGS=(--module "$MODULE")
if [[ "$RETENTION" -eq 1 ]]; then
    ARGS+=(--retention)
fi
set +e
python test_climatezones_data_exploration.py "${ARGS[@]}"
PYTHON_EXIT_CODE=$?
set -e

if [[ "$RETENTION" -eq 1 ]]; then
    ARTEFACT_DIR="$(<latest_artefact_dir.txt)"

    printf "%s\n" "$ENVIRONMENT_INVENTORY" \
        > "$ARTEFACT_DIR/environment_inventory.txt"

    printf "%s\n" "$ENVIRONMENT_HASH" \
        > "$ARTEFACT_DIR/environment_hash.txt"
fi

exit "$PYTHON_EXIT_CODE"
