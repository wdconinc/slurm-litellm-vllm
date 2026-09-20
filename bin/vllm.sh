#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --account=def-wdconinc
#SBATCH --partition=lgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=vllm_%j.log

# Load Singularity
module load singularity

# Load environment variables from .env if present
if [ -f "${SLURM_SUBMIT_DIR}/.env" ]; then
    export $(grep -v '^#' "${SLURM_SUBMIT_DIR}/.env" | xargs)
fi

# Pass Hugging Face Token to Singularity if defined
if [ -n "$HF_TOKEN" ]; then
    export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"
fi

# Automatically activate virtual environment if it exists
if [ -f "${SLURM_SUBMIT_DIR}/.venv/bin/activate" ]; then
    source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
fi

# Get the requested model from the first argument (default to mistral)
MODEL_KEY=${1:-mistral}

# Execute the Python Orchestrator
python -u "${SLURM_SUBMIT_DIR}/src/orchestrator.py" "$MODEL_KEY"
