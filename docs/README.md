# Slurm LiteLLM vLLM

Welcome to the documentation for **Slurm LiteLLM vLLM**. This project provides a robust, automated way to start up large language models on an HPC Slurm cluster (such as UManitoba's Grex) using vLLM in a Singularity container, and exposing them efficiently via a LiteLLM proxy on the login node.

## Architecture

This project is built around two primary environments: the **Login Node** and the **Compute Node**.

```mermaid
flowchart LR
    User([Local User / App])
    
    subgraph "HPC Cluster"
        subgraph "Login Node"
            SSH_Tunnel[SSH Port Forwarding\nLocal:4000 -> Login:4000]
            StartScript(bin/start.sh)
            LiteLLM[LiteLLM Proxy\n127.0.0.1:4000]
            Config[(dynamic_litellm_config.yaml)]
        end
        
        subgraph "Compute Node (GPU)"
            vLLMScript(bin/vllm.sh)
            Singularity[Singularity Container]
            vLLM[vLLM Server\n0.0.0.0:8000]
            Watchdog[Idle Timeout Watchdog]
        end
    end

    User -- SSH --> SSH_Tunnel
    SSH_Tunnel --> LiteLLM
    
    StartScript -- "1. sbatch" --> vLLMScript
    vLLMScript -- "2. Generate" --> Config
    StartScript -- "4. Read & Launch" --> LiteLLM
    LiteLLM -- "Read Config" --> Config
    
    vLLMScript -- "3. Start" --> Singularity
    Singularity --> vLLM
    vLLM -- "Metrics" --> Watchdog
    
    LiteLLM -- "Proxy API Requests" --> vLLM
    
    Watchdog -- "Kill on Idle" --> vLLM
```

### Components

1. **Start Script (`bin/start.sh`)**: The main entry point. It submits the slurm job, monitors the queue for initialization, and launches LiteLLM.
2. **vLLM Slurm Job (`bin/vllm.sh`)**: Executes on the compute node. It pulls the specified Hugging Face model from the cluster's parallel filesystem, starts the vLLM server via Singularity, and writes back the dynamically assigned compute node IP to a YAML configuration file.
3. **Idle Timeout Watchdog**: An integrated loop within the slurm script that monitors active connections to vLLM. If the server remains idle for 15 minutes, it terminates the process, relinquishing the GPU resources back to the Slurm scheduler.
4. **LiteLLM Proxy**: A lightweight server that maps the internal compute node's endpoint into a local `localhost:4000` port, standardizing the format to the OpenAI API specification.

## Usage Guide
Refer to the repository's main `README.md` for quickstart setup instructions and connection testing.
