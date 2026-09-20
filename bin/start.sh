#!/bin/bash
# Wrapper script to execute the new Python login node submitter

# Automatically activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

python bin/start.py "$@"
