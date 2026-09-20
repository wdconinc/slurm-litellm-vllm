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

# Get the absolute path to where you saved the model
#MODEL_NAME="qwen3-coder-next"
#MODEL_FULLNAME="Qwen/Qwen3-Coder-Next"
MODEL_NAME="mistralai-leanstral"
MODEL_FULLNAME="sahilchachra/Leanstral-1.5-119B-A6B-NVFP4"
MODEL_PATH="/project/6041615/models/${MODEL_NAME}"

# Run vLLM using Singularity. We bind to 0.0.0.0 so the login node can reach it.
#singularity exec --cleanenv --nv --bind /project/6041615 docker://vllm/vllm-openai:latest \
#    vllm serve $MODEL_PATH \
#    --served-model-name $MODEL_FULLNAME \
#    --host 0.0.0.0 \
#    --port 8000 \
#    --tensor-parallel-size 2 \
#    --enable-auto-tool-choice \
#    --tool-call-parser qwen3_xml \
#    --max-model-len 131072 \
#    --gpu-memory-utilization 0.95 \
#    --quantization fp8 \
#    --enable-prefix-caching \
#    --enable-chunked-prefill \
#    --trust-remote-code &
singularity exec --cleanenv --nv --bind /project/6041615 docker://vllm/vllm-openai:latest \
    vllm serve $MODEL_PATH \
    --served-model-name $MODEL_FULLNAME \
    --host 0.0.0.0 \
    --port 8000 \
    --limit-mm-per-prompt '{"image": 0}' \
    --tensor-parallel-size 2 \
    --quantization compressed-tensors \
    --max-model-len 32764 \
    --gpu-memory-utilization 0.90 \
    --tokenizer-mode mistral \
    --tool-call-parser mistral \
    --reasoning-parser mistral &

# Capture the Process ID of vLLM
VLLM_PID=$!

# Wait for the API to boot up, but exit if the process dies
echo "Waiting for vLLM to initialize..."
while ! curl -s http://localhost:8000/health > /dev/null; do
    # CRITICAL: Check if vLLM crashed while starting up
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM process died during initialization. Exiting job."
        exit 1
    fi
    sleep 5
done

echo "vLLM is up and running!"

# Grab the internal 10.x.x.x IP address of the compute node
INTERNAL_IP=$(hostname -I | awk '{for(i=1;i<=NF;i++) if($i ~ /^10\./) print $i}')
echo "Directing LiteLLM to internal compute IP: $INTERNAL_IP"

mkdir -p ~/litellm/etc
cat <<EOF > ~/litellm/etc/dynamic_litellm_config.yaml
model_list:
  - model_name: my-local-model
    litellm_params:
      model: openai/${MODEL_FULLNAME}
      api_base: http://${INTERNAL_IP}:8000/v1
      api_key: "not-needed"
EOF
echo "LiteLLM config successfully generated at ~/litellm/etc/dynamic_litellm_config.yaml"

# The Idle Timeout Watchdog
MAX_IDLE_MINUTES=15
IDLE_COUNTER=0

while true; do
    sleep 60 # Check every 60 seconds
    
    # Break if vLLM crashed or was killed externally
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM process stopped unexpectedly."
        break
    fi

    # Fetch metrics and sum up running, waiting, and swapped requests
    METRICS=$(curl -s http://localhost:8000/metrics)
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
