# System Architecture

The Slurm + vLLM infrastructure uses a **"Thin Bash, Fat Python"** architecture. While Slurm is fundamentally designed around Bash scripts containing `#SBATCH` directives, this project delegates all complex state management, process orchestration, and configuration generation to Python.

## Core Components

### 1. The Compute Node Orchestrator (`src/orchestrator.py`)
This daemon runs on the Slurm compute node and replaces complex shell-script logic.
- **Configuration Management**: Parses `config/models.yaml` to dynamically build vLLM and LiteLLM configurations.
- **Ray Cluster Initialization**: Uses `subprocess.Popen` to initialize the Ray head and worker nodes for multi-node deployments.
- **Health Monitoring**: Boots the vLLM Singularity container and waits for the `/health` endpoint to return HTTP 200 before exposing the proxy.
- **Idle Timeout Watchdog**: A threaded daemon that polls `http://127.0.0.1:8000/metrics` to track active requests. It gracefully sends `SIGTERM` to the container when the idle threshold is reached to prevent wasting GPU allocations.
- **Graceful Teardowns**: Uses `atexit` handlers to guarantee `endpoint.env` is safely deleted and Singularity zombie processes are fully reaped when Slurm preempts a job.

### 2. The Login Node Submitter (`bin/start.py`)
This script acts as the CLI entrypoint on the login node.
- **Dynamic Resource Allocation**: Reads `config/models.yaml` and injects `sbatch` arguments dynamically (e.g., requesting `--partition=skylake` and `--gpus-per-node=0` for CPU models, bypassing the default `#SBATCH` headers).
- **Progress Monitoring**: Uses `squeue` to fetch state data and provides a terminal progress UI while waiting for the compute node to initialize.

### 3. Programmatic API (`src/api.py`)
Researchers can import `SlurmLLMFleet` into Python notebooks (like Jupyter) to orchestrate multi-node LLM backends without touching the terminal. 
This allows automated integration testing, queue-based autoscaling, and dynamic model provisioning.

### 4. Configuration Schema (`config/models.yaml`)
Models are centrally defined using YAML. This cleanly separates infrastructure logic from specific model settings:

```yaml
mistral:
  fullname: "sahilchachra/Leanstral-1.5-119B-A6B-NVFP4"
  slurm:
    partition: "lgpu"
    nodes: 1
    gpus_per_node: 2
    cpus_per_task: 64
    mem: "128G"
    time: "04:00:00"
  vllm:
    args:
      - "--quantization compressed-tensors"
      - "--max-model-len 32764"
```
