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

# Get the hostname of the compute node we landed on
COMPUTE_NODE=$(hostname)
echo "vLLM is running on: $COMPUTE_NODE"

# Load Singularity
module load singularity

# Load environment variables from .env if present
if [ -f "${SLURM_SUBMIT_DIR}/.env" ]; then
    echo "Loading environment variables from ${SLURM_SUBMIT_DIR}/.env"
    export $(grep -v '^#' "${SLURM_SUBMIT_DIR}/.env" | xargs)
elif [ -f "$HOME/litellm/etc/.env" ]; then
    echo "Loading environment variables from $HOME/litellm/etc/.env"
    export $(grep -v '^#' "$HOME/litellm/etc/.env" | xargs)
fi

# Pass Hugging Face Token to Singularity if defined
if [ -n "$HF_TOKEN" ]; then
    export SINGULARITYENV_HF_TOKEN="$HF_TOKEN"
fi

# Get the requested model from the first argument (default to mistral)
MODEL_KEY=${1:-mistral}

case "$MODEL_KEY" in
    qwen|qwen3)
        MODEL_NAME="qwen3-coder-next"
        MODEL_FULLNAME="Qwen/Qwen3-Coder-Next"
        VLLM_ARGS=(
            "--enable-auto-tool-choice"
            "--tool-call-parser" "qwen3_xml"
            "--max-model-len" "131072"
            "--gpu-memory-utilization" "0.95"
            "--quantization" "fp8"
            "--enable-prefix-caching"
            "--enable-chunked-prefill"
            "--trust-remote-code"
        )
        ;;
    mistral|leanstral)
        MODEL_NAME="mistralai-leanstral"
        MODEL_FULLNAME="sahilchachra/Leanstral-1.5-119B-A6B-NVFP4"
        VLLM_ARGS=(
            "--limit-mm-per-prompt" '{"image": 0}'
            "--quantization" "compressed-tensors"
            "--max-model-len" "32764"
            "--gpu-memory-utilization" "0.90"
            "--tokenizer-mode" "mistral"
            "--tool-call-parser" "mistral"
            "--reasoning-parser" "mistral"
        )
        ;;
    *)
        echo "Unknown model key: $MODEL_KEY"
        echo "Available options: qwen, mistral"
        exit 1
        ;;
esac

# Optimize Startup and Downloading
# Enable ultra-fast Rust-based downloads from HuggingFace
export SINGULARITYENV_HF_HUB_ENABLE_HF_TRANSFER=1

# Point all caching mechanisms to the fast parallel filesystem instead of the login node's home dir
export SINGULARITYENV_HF_HOME="/project/6041615/huggingface_cache"
export SINGULARITYENV_TRITON_CACHE_DIR="/project/6041615/triton_cache"
export SINGULARITYENV_VLLM_CACHE_ROOT="/project/6041615/vllm_cache"

# Run vLLM using Singularity. We bind to 127.0.0.1 to force routing through LiteLLM.
# By passing $MODEL_FULLNAME instead of a local path, vLLM automatically downloads it
# (using the ultra-fast hf_transfer) into the shared --download-dir if it doesn't exist.
singularity exec --cleanenv --nv --bind /project/6041615 docker://vllm/vllm-openai:latest \
    vllm serve "$MODEL_FULLNAME" \
    --download-dir "/project/6041615/models" \
    --served-model-name "$MODEL_FULLNAME" \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 2 \
    "${VLLM_ARGS[@]}" &

# Capture the Process ID of vLLM
VLLM_PID=$!

# Wait for the API to boot up
echo "Waiting for vLLM to initialize..."
while ! curl -s http://127.0.0.1:8000/health > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died during initialization. Exiting job."
        exit 1
    fi
    sleep 5
done

echo "vLLM is up and running!"

# Grab the internal 10.x.x.x IP address of the compute node
INTERNAL_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i ~ /^10\./) print $i}')
if [ -z "$INTERNAL_IP" ]; then
    INTERNAL_IP=$(hostname -I | awk '{print $1}') # Fallback
fi
echo "Compute Node Internal IP: $INTERNAL_IP"

# Configure LiteLLM to point to local vLLM
mkdir -p ~/litellm/etc
cat <<EOF > ~/litellm/etc/dynamic_litellm_config_${SLURM_JOB_ID}.yaml
model_list:
  - model_name: my-local-model
    litellm_params:
      model: openai/${MODEL_FULLNAME}
      api_base: http://127.0.0.1:8000/v1
      api_key: "not-needed"
EOF

# Start LiteLLM proxy directly on this compute node
# Use the key from .env, or fallback to the default secret
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-hpc-secret-key}"
export OPENAI_API_KEY="not-needed"

# Automatically activate virtual environment if it exists
if [ -f "${SLURM_SUBMIT_DIR}/.venv/bin/activate" ]; then
    source "${SLURM_SUBMIT_DIR}/.venv/bin/activate"
fi

# Determine the correct litellm binary
LITELLM_CMD="litellm"
if ! command -v litellm &> /dev/null && [ -f "$HOME/litellm/bin/litellm" ]; then
    LITELLM_CMD="$HOME/litellm/bin/litellm"
fi

$LITELLM_CMD --config ~/litellm/etc/dynamic_litellm_config_${SLURM_JOB_ID}.yaml --host 0.0.0.0 --port 4000 &
LITELLM_PID=$!

# Publish the endpoint for workers
ENDPOINT_FILE="$HOME/litellm/etc/endpoint.env"
echo "export OPENAI_API_BASE=\"http://${INTERNAL_IP}:4000/v1\"" > "$ENDPOINT_FILE"
echo "export OPENAI_API_KEY=\"${LITELLM_MASTER_KEY}\"" >> "$ENDPOINT_FILE"
echo "export OPENAI_MODEL=\"my-local-model\"" >> "$ENDPOINT_FILE"
echo "LiteLLM endpoint published to $ENDPOINT_FILE"

# The Idle Timeout Watchdog
MAX_IDLE_MINUTES=15
IDLE_COUNTER=0

while true; do
    sleep 60 # Check every 60 seconds
    
    # Break if vLLM or LiteLLM crashed
    if ! kill -0 $VLLM_PID 2>/dev/null || ! kill -0 $LITELLM_PID 2>/dev/null; then
        echo "Critical server process stopped unexpectedly."
        break
    fi

    # Fetch metrics and calculate active requests robustly
    METRICS=$(curl -s http://127.0.0.1:8000/metrics || echo "CURL_FAILED")
    
    if [ "$METRICS" == "CURL_FAILED" ]; then
        echo "Warning: Metrics curl timed out. Assuming server is busy under load."
        IDLE_COUNTER=0
        continue
    fi
    ACTIVE_REQS=$(echo "$METRICS" | grep -E 'vllm:num_requests_(running|waiting|swapped)' | grep -v '#' | awk '{sum+=$2} END {print sum}')
    
    # Check if ACTIVE_REQS is exactly 0 (or empty, just in case)
    if [[ "$ACTIVE_REQS" == "0" ]] || [[ -z "$ACTIVE_REQS" ]]; then
        IDLE_COUNTER=$((IDLE_COUNTER + 1))
        echo "Server idle for $IDLE_COUNTER minute(s)..."
        
        if [ "$IDLE_COUNTER" -ge "$MAX_IDLE_MINUTES" ]; then
            echo "Max idle time ($MAX_IDLE_MINUTES mins) reached. Terminating vLLM to release GPU..."
            kill -15 $VLLM_PID
            wait $VLLM_PID
            echo "Job complete."
            break
        fi
    else
        # If there is activity, reset the counter
        if [ "$IDLE_COUNTER" -gt 0 ]; then
            echo "New request received. Resetting idle timer."
        fi
        IDLE_COUNTER=0
    fi
done
