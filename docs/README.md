# Slurm LiteLLM vLLM

Welcome to the documentation for **Slurm LiteLLM vLLM**. This project provides a robust, automated way to start up large language models on an HPC Slurm cluster (such as UManitoba's Grex) using vLLM in a Singularity container, and exposing them efficiently via a LiteLLM proxy on the login node.

## Architecture

This project is built around two primary environments: the **Login Node** and the **Compute Node**.

```mermaid
flowchart LR
    User([Local User / App])
    
    subgraph "HPC Cluster"
        subgraph "Login Node"
            SSH_Tunnel["SSH Port Forwarding<br/>Local:4000 -> Compute:4000"]
            StartScript("bin/start.sh")
            EndpointEnv[("endpoint.env")]
        end
        
        subgraph "Compute Node (GPU)"
            vLLMScript("bin/vllm.sh")
            LiteLLM["LiteLLM Proxy<br/>0.0.0.0:4000"]
            Config[("dynamic_litellm_config.yaml")]
            Singularity["Singularity Container"]
            vLLM["vLLM Server<br/>127.0.0.1:8000"]
            Watchdog["Idle Timeout Watchdog"]
        end
    end

    User -- SSH --> SSH_Tunnel
    SSH_Tunnel --> LiteLLM
    
    StartScript -- "1. sbatch" --> vLLMScript
    vLLMScript -- "2. Generate" --> Config
    vLLMScript -- "3. Publish" --> EndpointEnv
    StartScript -- "4. Wait & Read" --> EndpointEnv
    LiteLLM -- "Read Config" --> Config
    
    vLLMScript -- "5. Start Proxy" --> LiteLLM
    vLLMScript -- "6. Start vLLM" --> Singularity
    Singularity --> vLLM
    vLLM -- "Metrics" --> Watchdog
    
    LiteLLM -- "Proxy API Requests" --> vLLM
    
    Watchdog -- "Kill on Idle" --> vLLM
```

### Startup Sequence

The diagram below illustrates the exact chronological sequence of events when a user starts the system:

```mermaid
sequenceDiagram
    participant User as User / App
    participant Login as Login Node
    participant Slurm as Slurm Scheduler
    participant GPU as GPU Compute Node
    
    User->>Login: Run ./bin/start.sh
    Login->>Slurm: sbatch bin/vllm.sh
    Slurm-->>Login: Return Job ID
    
    Note over Login: Polls endpoint.env<br/>waiting for creation
    
    Slurm->>GPU: Allocate node & execute vllm.sh
    GPU->>GPU: Download/Cache Model (hf_transfer)
    GPU->>GPU: Start vLLM Server (Singularity)
    
    Note over GPU: Wait for http://127.0.0.1:8000/health
    
    GPU->>GPU: Start LiteLLM Proxy (0.0.0.0:4000)
    GPU->>Login: Write IP & Keys to endpoint.env
    
    Login-->>User: Print connection instructions & exit
    
    Note over User,GPU: Runtime Execution
    User->>GPU: POST /v1/chat/completions<br/>(Routed via SSH Tunnel / Network)
    GPU->>GPU: LiteLLM processes & forwards to vLLM
    GPU-->>User: JSON Response
```

### Components

1. **Start Script (`bin/start.sh`)**: The main entry point on the login node. It submits the slurm job and waits for the `endpoint.env` file to be populated by the compute node, ensuring you know exactly where to tunnel.
2. **vLLM Slurm Job (`bin/vllm.sh`)**: Executes on the compute node. It starts the vLLM server via Singularity, dynamically generates the LiteLLM config, launches the LiteLLM proxy in the background, and writes back the compute node's internal IP to `endpoint.env`.
3. **Idle Timeout Watchdog**: An integrated loop within the slurm script that monitors active connections to vLLM. If the server remains idle for 15 minutes, it terminates the process, relinquishing the GPU resources back to the Slurm scheduler.
4. **LiteLLM Proxy**: A lightweight proxy now bundled on the compute node. It maps the internal vLLM endpoint (`127.0.0.1:8000`) into a cluster-accessible standard OpenAI API interface (`0.0.0.0:4000`).

## Usage Guide
Refer to the repository's main `README.md` for quickstart setup instructions and connection testing.
