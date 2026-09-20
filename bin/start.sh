#!/bin/bash

# Default model key
MODEL_KEY=${1:-mistral}

CONFIG_FILE="$HOME/litellm/etc/dynamic_litellm_config.yaml"

echo "Submitting vLLM Slurm job for model: $MODEL_KEY..."

# Remove old config to ensure we wait for the new one
rm -f "$CONFIG_FILE"

# Submit the job
OUTPUT=$(sbatch bin/vllm.sh "$MODEL_KEY" 2>&1)
if [ $? -ne 0 ]; then
    echo "Error: Failed to submit Slurm job."
    echo "$OUTPUT"
    exit 1
fi

echo "$OUTPUT"
# Extract job ID assuming output format like "Submitted batch job 123456"
JOB_ID=$(echo "$OUTPUT" | awk '{print $4}')

if [[ -z "$JOB_ID" || ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "Warning: Could not parse job ID from sbatch output. Will wait for config file indefinitely."
else
    echo "Waiting for Slurm job $JOB_ID to initialize and generate LiteLLM config..."
    echo "(This may take a few minutes while the container starts and the model loads)"
fi

# Wait until the config file is generated
while [ ! -f "$CONFIG_FILE" ]; do
    if [[ -n "$JOB_ID" && "$JOB_ID" =~ ^[0-9]+$ ]]; then
        # Check if job is still in squeue. If it fails, squeue won't have it.
        # squeue might take a second to register the job, so we wait 5 seconds before the loop repeats
        if ! squeue -h -j "$JOB_ID" 2>/dev/null | grep -q "$JOB_ID"; then
            echo "Error: Slurm job $JOB_ID is no longer in the queue. It may have failed during initialization."
            echo "Please check vllm_${JOB_ID}.log for details."
            exit 1
        fi
    fi
    sleep 5
done

echo "Configuration generated! Starting LiteLLM proxy..."

# Optional but recommended: set a master key to keep other login node users out
export LITELLM_MASTER_KEY="sk-hpc-secret-key" 
export OPENAI_API_KEY="not-needed"

$HOME/litellm/bin/litellm --config "$CONFIG_FILE" --host 127.0.0.1 --port 4000
