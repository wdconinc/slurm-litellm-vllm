# Slurm LiteLLM vLLM

This repository contains scripts to start an automatically expiring vLLM instance on a Slurm HPC cluster (specifically configured for the Grex cluster at UManitoba) and proxy the local LLM on the login node using LiteLLM. This setup allows you to efficiently use GPU resources on compute nodes while providing a standard OpenAI-compatible API endpoint that you can access over an SSH port-forwarded connection.

## Features

- **Automated Resource Management**: Includes a 15-minute idle timeout watchdog that monitors active requests. If the vLLM server is idle for 15 minutes, the Slurm job automatically terminates to release the allocated GPUs.
- **Dynamic Networking**: Automatically determines the internal IP address of the compute node running vLLM and generates a configuration file (`dynamic_litellm_config.yaml`) for the LiteLLM proxy on the login node.
- **Singularity Integration**: Runs the vLLM server inside a Singularity container directly from the `vllm-openai` docker image.

## Prerequisites

- Access to a Slurm cluster (e.g., UManitoba Grex).
- **Singularity** module available on compute nodes.
- Local model weights stored in a directory accessible to the compute nodes (default is `/project/6041615/models/`).
- Python environment with the dependencies in `requirements.txt` installed (primarily for LiteLLM).

## Usage

### 1. Submit the vLLM Slurm Job

From the login node, submit the job to request a GPU node and start the vLLM container. You can pass the model key as an argument (defaults to `mistral`):

```bash
sbatch bin/vllm.sh mistral   # or 'qwen'
```

This script will:
1. Request 1 node with 2 GPUs and 128GB of memory for up to 4 hours.
2. Boot up vLLM with the selected model (e.g., `Leanstral-1.5-119B-A6B-NVFP4` or `Qwen3-Coder-Next`) and its specific optimized parameters.
3. Generate the proxy configuration file at `~/litellm/etc/dynamic_litellm_config.yaml`.
4. Run the watchdog to monitor idle time.

### 2. Start the LiteLLM Proxy

Once the Slurm job is running and has generated the configuration file, start the LiteLLM proxy on the login node:

```bash
./bin/litellm.sh
```

By default, the proxy runs on `127.0.0.1:4000` on the login node.

*(Optional: You can change the `LITELLM_MASTER_KEY` in `bin/litellm.sh` to secure your proxy endpoint).*

### 3. Connect via SSH Port Forwarding

To reliably access the LiteLLM proxy from your local machine, it is recommended to configure your `~/.ssh/config` file. This handles the port forwarding automatically and ensures your connection stays alive:

```ssh-config
Host *.hpc.umanitoba.ca
    IdentityFile ~/.ssh/id_rsa.grex.hpc.umanitoba.ca  # Update with your actual key path
    LocalForward 4000 127.0.0.1:4000
    ServerAliveInterval 60
```

With this configured, simply SSH into the login node:

```bash
ssh your_username@grex.hpc.umanitoba.ca
```

*(Alternatively, for a one-off connection without editing your config, you can run: `ssh -L 4000:127.0.0.1:4000 your_username@grex.hpc.umanitoba.ca`)*

You can now use your local `localhost:4000` as an OpenAI-compatible API base in your applications, scripts, or IDEs:

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-hpc-secret-key" # Must match LITELLM_MASTER_KEY in litellm.sh
)

response = client.chat.completions.create(
    model="my-local-model",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Adding New Models

The application now natively supports multiple models via a `case` statement in `bin/vllm.sh`. The default supported models are `mistral` and `qwen`.

To add a new model:
1. Open `bin/vllm.sh`.
2. Locate the `case "$MODEL_KEY" in` section.
3. Add a new switch case for your model (e.g., `llama3)`).
4. Define `MODEL_NAME` (directory name in `/project/6041615/models/`) and `MODEL_FULLNAME` (Hugging Face identifier for the served model name).
5. Specify any custom arguments required for your model as an array in `VLLM_ARGS`.
6. Submit your job with `sbatch bin/vllm.sh llama3`.
