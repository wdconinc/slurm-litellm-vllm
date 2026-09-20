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
echo "Model: $OPENAI_MODEL"
echo "------------------------------------------------------"

# Run the pytest suite
pytest -v tests/
