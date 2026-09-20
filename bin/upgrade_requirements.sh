#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================"
echo " Requirements Upgrade & Validation Script"
echo "======================================================"

# 1. Ensure we have a base list of packages without strict version pins
BASE_REQS="requirements.in"
if [ ! -f "$BASE_REQS" ]; then
    echo "Extracting base packages from requirements.txt..."
    # Strip versions (everything after ==, >=, <=, etc.)
    sed -E 's/([a-zA-Z0-9_-]+).*/\1/' requirements.txt > "$BASE_REQS"
fi

# 2. Setup a sandboxed test environment
TEST_VENV=".venv_upgrade_test"
echo "Creating isolated test environment in $TEST_VENV..."
rm -rf "$TEST_VENV"
python -m venv "$TEST_VENV"
source "$TEST_VENV/bin/activate"

# 3. Install latest versions
echo "Upgrading pip and installing latest package versions..."
pip install --upgrade pip > /dev/null
pip install --upgrade -r "$BASE_REQS"

# 4. Validate against the live LLM endpoint
ENDPOINT_FILE="$HOME/litellm/etc/endpoint.env"
if [ ! -f "$ENDPOINT_FILE" ]; then
    echo ""
    echo "⚠️ WARNING: Cannot validate performance!"
    echo "No active proxy endpoint found at $ENDPOINT_FILE."
    echo "Please start the Slurm job (./bin/start.sh) and wait for the proxy to initialize before upgrading."
    deactivate
    rm -rf "$TEST_VENV"
    exit 1
fi

source "$ENDPOINT_FILE"

echo "Running validation test against active proxy..."
echo "------------------------------------------------------"

# Test the upgraded OpenAI client against the active LiteLLM proxy
# Using Atomic Number 42 (Molybdenum) as a test prompt
if python examples/query_element.py 42; then
    echo "------------------------------------------------------"
    echo "✅ Validation Passed: Upgraded packages successfully communicated with the proxy."
    
    # 5. Freeze the successful upgrade
    echo "Freezing new pinned versions to requirements.txt..."
    pip freeze > requirements.txt
    
    echo "Success! Your requirements.txt has been permanently updated."
    echo ""
    echo "Next steps:"
    echo "  1. Run 'git diff requirements.txt' to review the updates."
    echo "  2. Sync your main environment: 'source .venv/bin/activate && pip install -r requirements.txt'"
else
    echo "------------------------------------------------------"
    echo "❌ Validation Failed: The upgraded packages encountered an error."
    echo "Your requirements.txt has NOT been modified."
fi

# Cleanup
echo "Cleaning up test environment..."
deactivate
rm -rf "$TEST_VENV"
