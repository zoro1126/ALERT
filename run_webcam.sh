#!/usr/bin/env bash
set -euo pipefail

# Determine repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Locate Python binary (favor local virtual environment)
if [ -x ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
elif [ -x ".venv/bin/python3" ]; then
    PYTHON_EXEC=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXEC="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXEC="python"
else
    echo "Error: Python interpreter not found." >&2
    exit 1
fi

export PYTHONPATH="src:${PYTHONPATH:-}"

echo "========================================================"
echo "  ALERT Driver Fatigue Detection System (Production Model)"
echo "========================================================"
echo "Starting camera stream..."

set +e
"$PYTHON_EXEC" src/run_alert.py "$@"
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -ne 0 ]; then
    echo ""
    echo "An error occurred while running ALERT (exit code: $EXIT_CODE)."
    exit "$EXIT_CODE"
fi
