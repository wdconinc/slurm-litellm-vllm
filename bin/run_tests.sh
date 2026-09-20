#!/bin/bash
set -e

echo "======================================================"
echo " Proxy Feature Verification Suite"
echo "======================================================"

ENDPOINT_FILE="$HOME/litellm/etc/endpoint.env"
if [ ! -f "$ENDPOINT_FILE" ]; then
    echo "❌ ERROR: Proxy endpoint not found at $ENDPOINT_FILE"
    echo "Please start the server first using ./bin/start.sh and wait for initialization."
    exit 1
fi

source "$ENDPOINT_FILE"

# Activate local venv if present
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

# Ensure pytest is installed
if ! python -c "import pytest" &> /dev/null; then
    echo "Installing pytest..."
    pip install pytest > /dev/null
fi

echo "Running full feature verification against $OPENAI_API_BASE..."
# Check for integration flag
if [ "$1" == "--run-integration" ]; then
    export RUN_RAY_INTEGRATION_TEST=1
    echo "⚠️  Integration testing ENABLED."
    echo "This will actively submit a 2-node Slurm job, which may take 10+ minutes to run."
else
    echo "ℹ️  Skipping slow Ray multi-node integration test."
    echo "Run with './bin/run_tests.sh --run-integration' to include it."
fi

echo "------------------------------------------------------"

# Run the pytest suite
pytest -v tests/
