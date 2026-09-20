import os
import sys
import time
import socket
import subprocess
import signal
import yaml
import requests
import atexit

# Global list of child processes to clean up
PROCESSES = []
ENDPOINT_FILE = os.path.expanduser("~/litellm/etc/endpoint.env")

def cleanup():
    print("\n[Orchestrator] Cleaning up resources...")
    if os.path.exists(ENDPOINT_FILE):
        os.remove(ENDPOINT_FILE)
    for p in PROCESSES:
        if p.poll() is None:
            p.terminate()
            p.wait()

atexit.register(cleanup)

def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_internal_ip():
    try:
        ips = subprocess.check_output(["hostname", "-I"], text=True).strip().split()
        for ip in ips:
            if ip.startswith("10."):
                return ip
        return ips[0] if ips else "127.0.0.1"
    except Exception:
        return "127.0.0.1"

def deep_merge(dict1, dict2):
    for k, v in dict2.items():
        if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
            deep_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1

def get_model_config(model_key):
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml")
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading config/models.yaml: {e}")
        sys.exit(1)
        
    if model_key in data and isinstance(data[model_key], str):
        model_key = data[model_key]
        
    if model_key not in data:
        print(f"Unknown model key: {model_key}")
        sys.exit(1)
        
    defaults = data.get("defaults", {})
    model_config = data[model_key]
    
    import copy
    config = copy.deepcopy(defaults)
    deep_merge(config, model_config)
    
    return config

def main():
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <model_key>")
        sys.exit(1)
        
    model_key = sys.argv[1]
    config = get_model_config(model_key)
    
    # Environment Setup
    os.environ["SINGULARITYENV_HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    os.environ["SINGULARITYENV_HF_HOME"] = "/project/6041615/huggingface_cache"
    os.environ["SINGULARITYENV_TRITON_CACHE_DIR"] = "/project/6041615/triton_cache"
    os.environ["SINGULARITYENV_VLLM_CACHE_ROOT"] = "/project/6041615/vllm_cache"
    
    os.environ["HF_HOME"] = "/project/6041615/huggingface_cache"
    os.environ["TIKTOKEN_CACHE_DIR"] = "/project/6041615/tiktoken_cache"
    
    internal_ip = get_internal_ip()
    compute_node = socket.gethostname()
    print(f"[Orchestrator] Running on {compute_node} (IP: {internal_ip})")
    
    num_nodes = int(os.getenv("SLURM_JOB_NUM_NODES", "1"))
    gpus_per_node = int(os.getenv("SLURM_GPUS_PER_NODE", "2"))
    
    vllm_image = config.get("vllm", {}).get("image", "docker://vllm/vllm-openai:v0.6.3.post1")
    sing_bind_path = config.get("vllm", {}).get("sing_bind", "/project/6041615")
    vllm_host = config.get("vllm", {}).get("host", "127.0.0.1")
    vllm_port = str(config.get("vllm", {}).get("port", "8000"))
    vllm_download_dir = config.get("vllm", {}).get("download_dir", "/project/6041615/models")
    
    if gpus_per_node == 0 or "cpu" in model_key:
        print("[Orchestrator] Running in CPU-only mode.")
        sing_bind = ["--bind", sing_bind_path]
        tp_size = num_nodes
    else:
        sing_bind = ["--nv", "--bind", sing_bind_path]
        tp_size = num_nodes * gpus_per_node

    import shlex
    raw_vllm_args = config.get("vllm", {}).get("args", [])
    vllm_args = []
    for arg in raw_vllm_args:
        vllm_args.extend(shlex.split(arg))
    
    # Ray Cluster Initialization
    if num_nodes > 1:
        print(f"[Orchestrator] Multi-node setup detected ({num_nodes} nodes). Configuring Ray...")
        ray_port = "6379"
        ip_head = f"{internal_ip}:{ray_port}"
        os.environ["SINGULARITYENV_RAY_ADDRESS"] = ip_head
        
        cpus_per_task = os.getenv("SLURM_CPUS_PER_TASK", "64")
        
        # Head node
        head_cmd = ["srun", "--nodes=1", "--ntasks=1", "-w", compute_node, "singularity", "exec", "--cleanenv"] + sing_bind + [vllm_image, "ray", "start", "--head", f"--node-ip-address={internal_ip}", f"--port={ray_port}", f"--num-cpus={cpus_per_task}", "--block"]
        PROCESSES.append(subprocess.Popen(head_cmd))
        
        # Worker nodes
        if num_nodes - 1 > 0:
            worker_cmd = ["srun", f"--nodes={num_nodes-1}", f"--ntasks={num_nodes-1}", "--exclude", compute_node, "singularity", "exec", "--cleanenv"] + sing_bind + [vllm_image, "ray", "start", f"--address={ip_head}", f"--num-cpus={cpus_per_task}", "--block"]
            PROCESSES.append(subprocess.Popen(worker_cmd))
            
        time.sleep(10)
        vllm_args.append("--worker-use-ray")

    # Start vLLM
    print("[Orchestrator] Starting vLLM...")
    vllm_cmd = ["singularity", "exec", "--cleanenv"] + sing_bind + [
        vllm_image, "vllm", "serve", config["fullname"],
        "--download-dir", vllm_download_dir,
        "--served-model-name", config["fullname"],
        "--host", vllm_host, "--port", vllm_port,
        "--tensor-parallel-size", str(tp_size)
    ] + vllm_args
    
    vllm_proc = subprocess.Popen(vllm_cmd)
    PROCESSES.append(vllm_proc)
    
    # Wait for vLLM health
    print("[Orchestrator] Waiting for vLLM health check...")
    healthy = False
    while vllm_proc.poll() is None:
        try:
            r = requests.get(f"http://{vllm_host}:{vllm_port}/health", timeout=2)
            if r.status_code == 200:
                healthy = True
                break
        except Exception:
            pass
        time.sleep(5)
        
    if not healthy:
        print("[Orchestrator] vLLM failed to start.")
        sys.exit(1)
        
    print("[Orchestrator] vLLM is up and running!")
    
    # LiteLLM Configuration
    job_id = os.getenv("SLURM_JOB_ID", "local")
    os.makedirs(os.path.expanduser("~/litellm/etc"), exist_ok=True)
    yaml_path = os.path.expanduser(f"~/litellm/etc/dynamic_litellm_config_{job_id}.yaml")
    
    litellm_config = {
        "model_list": [
            {
                "model_name": "my-local-model",
                "litellm_params": {
                    "model": f"openai/{config['fullname']}",
                    "api_base": f"http://{vllm_host}:{vllm_port}/v1",
                    "api_key": "not-needed"
                }
            }
        ]
    }
    
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        litellm_config["litellm_settings"] = {"success_callbacks": ["langfuse"]}
        print("[Orchestrator] Langfuse observability enabled.")
        
    with open(yaml_path, "w") as f:
        yaml.dump(litellm_config, f)
        
    # Start LiteLLM
    print("[Orchestrator] Starting LiteLLM Proxy...")
    master_key = os.getenv("LITELLM_MASTER_KEY", "sk-hpc-secret-key")
    os.environ["LITELLM_MASTER_KEY"] = master_key
    os.environ["OPENAI_API_KEY"] = "not-needed"
    
    litellm_cmd = "litellm"
    if not subprocess.run(["command", "-v", "litellm"], capture_output=True, shell=True).returncode == 0:
        if os.path.exists(os.path.expanduser("~/litellm/bin/litellm")):
            litellm_cmd = os.path.expanduser("~/litellm/bin/litellm")
            
    litellm_port = str(config.get("litellm", {}).get("port", "4000"))
    proxy_proc = subprocess.Popen([litellm_cmd, "--config", yaml_path, "--host", "0.0.0.0", "--port", litellm_port])
    PROCESSES.append(proxy_proc)
    
    # Publish endpoint
    with open(ENDPOINT_FILE, "w") as f:
        f.write(f'export OPENAI_API_BASE="http://{internal_ip}:{litellm_port}/v1"\n')
        f.write(f'export OPENAI_API_KEY="{master_key}"\n')
        f.write('export OPENAI_MODEL="my-local-model"\n')
    print(f"[Orchestrator] Endpoint published to {ENDPOINT_FILE}")
    
    # Idle Watchdog
    max_idle_mins = int(config.get("watchdog", {}).get("max_idle_mins", 15))
    idle_mins = 0
    while vllm_proc.poll() is None and proxy_proc.poll() is None:
        time.sleep(60)
        try:
            r = requests.get(f"http://{vllm_host}:{vllm_port}/metrics", timeout=5)
            metrics = r.text
            active_reqs = 0
            for line in metrics.splitlines():
                if line.startswith("vllm:num_requests_") and not line.startswith("#"):
                    active_reqs += float(line.split()[1])
            
            if active_reqs == 0:
                idle_mins += 1
                print(f"[Watchdog] Server idle for {idle_mins} minute(s)...")
                if idle_mins >= max_idle_mins:
                    print(f"[Watchdog] Max idle time ({max_idle_mins} mins) reached. Terminating.")
                    break
            else:
                if idle_mins > 0:
                    print("[Watchdog] New request received. Resetting timer.")
                idle_mins = 0
        except Exception as e:
            print(f"[Watchdog] Error fetching metrics: {e}")
            idle_mins = 0

    print("[Orchestrator] Shutting down.")

if __name__ == "__main__":
    main()
