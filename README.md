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
- Python environment with the dependencies in `requirements.txt` installed (primarily for LiteLLM):

  ```bash
  git clone https://github.com/wdconinc/slurm-litellm-vllm.git
  cd slurm-litellm-vllm

  # Create and activate a virtual environment (highly recommended)
  python -m venv .venv
  source .venv/bin/activate

  # Install requirements
  pip install -r requirements.txt
  ```

## Usage

### 1. Start the vLLM Server and Proxy
**Via Terminal:**
From the login node, simply run the unified startup script. You can optionally pass the model key as an argument (defaults to `mistral`):

```bash
./bin/start.sh mistral   # or 'smollm-cpu'
```

**Via Python API (Jupyter Notebooks):**
```python
from src.api import SlurmLLMFleet

fleet = SlurmLLMFleet(model="smollm-cpu")
fleet.start(wait=True)
client = fleet.get_openai_client()
```

The system will automatically:
1. Parse `config/models.yaml` to request the correct Slurm resources (CPU/GPU nodes).
2. Wait for the job to initialize and generate the required proxy configuration.
3. Automatically launch the LiteLLM proxy on the login node (`127.0.0.1:4000`) as soon as the configuration is ready.

By default, the proxy runs on `127.0.0.1:4000` on the login node.

*(Optional: You can change the `LITELLM_MASTER_KEY` directly inside `.env` to secure your proxy endpoint).*

### 2. Connect via SSH Port Forwarding

To reliably access the endpoint from your local machine, use the provided local helper script. This script automatically reads the active configuration on the cluster and sets up the correct port forwarding directly to the active compute node:

```bash
./bin/connect.sh your_username@grex.hpc.umanitoba.ca
```

*(Alternatively, for a manual setup, you can check `~/litellm/etc/endpoint.env` on the login node and manually run: `ssh -L 4000:<INTERNAL_IP>:4000 your_username@grex.hpc.umanitoba.ca`)*

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

## Configuration and Secrets

To manage sensitive keys (like Hugging Face access tokens for gated models) and override default settings, you can create a `.env` file in the root of the repository:

```bash
cp .env.example .env
```

Open `.env` and configure your settings:
- `LITELLM_MASTER_KEY`: Protects your endpoint. Changing this replaces the default `sk-hpc-secret-key`.
- `HF_TOKEN`: Required if you are downloading gated Hugging Face models (like Llama-3).
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`: (Optional) Setting these enables Langfuse observability and telemetry in LiteLLM.

The `bin/vllm.sh` script automatically reads this `.env` file on startup and passes the necessary credentials securely into the Singularity container.

## Adding New Models

Model definitions and their specific HPC hardware requirements are centrally configured in a declarative YAML file located at **`config/models.yaml`**. 

To add a new model:
1. Open `config/models.yaml`.
2. Add a new YAML block for your model.
3. Define the `fullname` (the exact Hugging Face repository ID).
4. Define the `slurm` requirements (`partition`, `gpus_per_node`, `cpus_per_task`).
5. Specify any custom optimization arguments required for your model in the `vllm: args` list.

Example:
```yaml
llama3:
  fullname: "meta-llama/Meta-Llama-3-8B-Instruct"
  slurm:
    partition: "lgpu"
    nodes: 1
    gpus_per_node: 1
    cpus_per_task: 32
    mem: "64G"
    time: "02:00:00"
  vllm:
    args:
      - "--quantization fp8"
      - "--max-model-len 8192"
```

Once saved, simply run `./bin/start.sh llama3` or deploy it programmatically via Python.

*Note: You no longer need to download models manually. The infrastructure will automatically download missing models into `/project/6041615/models` using ultra-fast rust-based `hf_transfer`.*
