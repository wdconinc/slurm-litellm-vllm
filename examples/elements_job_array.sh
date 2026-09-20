#!/bin/bash
#SBATCH --job-name=elements_array
#SBATCH --array=1-20          # Launch 20 parallel jobs (atomic numbers 1 to 20)
#SBATCH --cpus-per-task=1     # Minimal CPU requirement
#SBATCH --mem=1G              # Minimal Memory requirement
#SBATCH --time=00:10:00       # Short duration
#SBATCH --output=element_job_%A_%a.log

# 1. Load the centralized LLM endpoint published by bin/start.sh
ENDPOINT_FILE="/home/wdconinc/git/slurm-litellm-vllm/run/endpoint.env"

if [ ! -f "$ENDPOINT_FILE" ]; then
    echo "ERROR: Endpoint file $ENDPOINT_FILE not found."
    echo "Please ensure you have launched the vLLM proxy via bin/start.sh first."
    exit 1
fi

# This populates OPENAI_API_BASE, OPENAI_API_KEY, and OPENAI_MODEL
source "$ENDPOINT_FILE"

# 2. Activate virtual environment if it exists
if [ -f "$SLURM_SUBMIT_DIR/.venv/bin/activate" ]; then
    source "$SLURM_SUBMIT_DIR/.venv/bin/activate"
fi

# 3. Run the Python script to query the LLM
# The Python openai library automatically detects the environment variables
echo "Worker ${SLURM_ARRAY_TASK_ID} starting..."

python examples/query_element.py ${SLURM_ARRAY_TASK_ID}

echo "Worker ${SLURM_ARRAY_TASK_ID} finished."
