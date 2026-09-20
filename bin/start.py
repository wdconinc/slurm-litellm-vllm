import os
import sys
import time
import subprocess

def deep_merge(dict1, dict2):
    for k, v in dict2.items():
        if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
            deep_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1

def submit_job(model_key):
    import yaml
    
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml")
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading config/models.yaml: {e}")
        sys.exit(1)
        
    if model_key in data and isinstance(data[model_key], str):
        model_key = data[model_key] # resolve alias
        
    if model_key not in data:
        print(f"Unknown model key: {model_key}")
        sys.exit(1)
        
    defaults = data.get("defaults", {})
    model_config = data[model_key]
    
    import copy
    config = copy.deepcopy(defaults)
    deep_merge(config, model_config)
    
    slurm_overrides = config.get("slurm", {})
    
    cmd = ["sbatch"]
    for key, val in slurm_overrides.items():
        # e.g., "gpus_per_node" -> "--gpus-per-node"
        arg = f"--{key.replace('_', '-')}"
        cmd.append(f"{arg}={val}")
        
    cmd.extend(["bin/vllm.sh", model_key])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Failed to submit Slurm job.")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
        
    stdout = result.stdout.strip()
    print(stdout)
    
    job_id = None
    for line in stdout.splitlines():
        if "Submitted batch job" in line:
            job_id = line.split()[-1]
            break
            
    if not job_id or not job_id.isdigit():
        print("Warning: Could not parse job ID from sbatch output. Will wait for config file indefinitely.")
    else:
        print(f"Waiting for Slurm job {job_id} to initialize and generate LiteLLM config...")
        print("(This may take a few minutes while the container starts and the model loads)")
        
    return job_id

def monitor_job(job_id):
    endpoint_file = os.path.join(os.path.dirname(__file__), "..", "run", "endpoint.env")
    if os.path.exists(endpoint_file):
        os.remove(endpoint_file)
        
    start_time = time.time()
    
    while not os.path.exists(endpoint_file):
        elapsed = int(time.time() - start_time)
        if job_id:
            squeue_res = subprocess.run(["squeue", "-h", "-j", job_id, "-o", "%T"], capture_output=True, text=True)
            status = squeue_res.stdout.strip()
            
            if not status:
                print(f"\n❌ Error: Slurm job {job_id} is no longer in the queue. It likely failed.")
                print(f"Please check vllm_{job_id}.log for details.")
                sys.exit(1)
                
            sys.stdout.write(f"\r⏳ Waiting for endpoint.env... Job {job_id} is [ {status} ] (Elapsed: {elapsed}s)   ")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r⏳ Waiting for endpoint.env... (Elapsed: {elapsed}s)   ")
            sys.stdout.flush()
            
        time.sleep(5)
        
    print("\n")
    return endpoint_file

def print_success(endpoint_file):
    with open(endpoint_file, "r") as f:
        content = f.read()
        
    api_base = None
    for line in content.splitlines():
        if "OPENAI_API_BASE" in line:
            api_base = line.split("=", 1)[1].strip('"')
            break
            
    if api_base:
        # OPENAI_API_BASE="http://10.X.X.X:4000/v1"
        compute_ip = api_base.split("//")[1].split(":")[0]
    else:
        compute_ip = "<COMPUTE_NODE_IP>"
        
    print("==========================================================")
    print("✅ Success! Decentralized LLM Proxy is running.")
    print("==========================================================")
    print("For Cluster Workers (e.g., Lean4 Fleet):")
    print("  Before running your worker scripts, load the endpoint:")
    print(f"  source {endpoint_file}")
    print("\nFor Local Laptop Access (SSH Port Forwarding):")
    print("  Run this command on your laptop to tunnel to the compute node:")
    print(f"  ssh -L 4000:{compute_ip}:4000 your_username@grex.hpc.umanitoba.ca")
    print("==========================================================")

def main():
    model_key = sys.argv[1] if len(sys.argv) > 1 else "mistral"
    print(f"Submitting vLLM Slurm job for model: {model_key}...")
    job_id = submit_job(model_key)
    endpoint_file = monitor_job(job_id)
    print_success(endpoint_file)

if __name__ == "__main__":
    main()
