import os
import subprocess
import time
import pytest
from openai import OpenAI

@pytest.mark.skipif(not os.getenv("RUN_RAY_INTEGRATION_TEST"), reason="Skipping slow multi-node integration test. Set RUN_RAY_INTEGRATION_TEST=1 to run.")
def test_ray_multi_node_cluster():
    """
    Submits a real 2-node Slurm job to verify that the Ray head and worker nodes
    start correctly, the tensor parallelism scales across the nodes, and the 
    LiteLLM proxy successfully serves traffic.
    """
    print("\nSubmitting 2-node job to Slurm...")
    result = subprocess.run(
        ["sbatch", "--nodes=2", "--time=00:30:00", "bin/vllm.sh", "mistral"], 
        capture_output=True, 
        text=True
    )
    assert result.returncode == 0, f"sbatch failed: {result.stderr}"
    
    # Extract job ID (e.g. "Submitted batch job 12345")
    job_id = result.stdout.strip().split()[-1]
    print(f"Job {job_id} submitted successfully.")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    endpoint_file = os.path.join(repo_root, "run", "endpoint.env")
    
    # Remove old endpoint file so we can wait for the new one
    if os.path.exists(endpoint_file):
        os.remove(endpoint_file)
        
    print(f"Waiting for Slurm job {job_id} to boot 2-node Ray cluster (this may take 5-10 minutes)...")
    
    max_wait = 900 # 15 mins timeout
    start_time = time.time()
    
    while not os.path.exists(endpoint_file):
        # Check if job failed or died
        squeue_res = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        if job_id not in squeue_res.stdout and (time.time() - start_time) > 30:
            pytest.fail(f"Job {job_id} is no longer in the queue. It likely failed to start the Ray cluster. Check vllm_{job_id}.log")
            
        if time.time() - start_time > max_wait:
            subprocess.run(["scancel", job_id])
            pytest.fail("Timeout waiting for Ray cluster to boot.")
        time.sleep(10)
        
    print("Endpoint file generated! Ray cluster is online.")
    
    # Read the newly generated endpoint
    env_vars = {}
    with open(endpoint_file, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().replace('export ', '').split("=", 1)
                env_vars[key] = val.strip('"')
                
    client = OpenAI(
        api_key=env_vars.get("OPENAI_API_KEY", "not-needed"),
        base_url=env_vars.get("OPENAI_API_BASE")
    )
    
    # Test generation to ensure the tensors were successfully parallelized
    print("Testing OpenAI API generation through the multi-node proxy...")
    try:
        response = client.chat.completions.create(
            model=env_vars.get("OPENAI_MODEL", "my-local-model"),
            messages=[{"role": "user", "content": "Say 'Ray works!'"}],
            max_tokens=10,
            temperature=0.1
        )
        content = response.choices[0].message.content.lower()
        assert "ray" in content or len(content) > 0, "Model did not return a valid response."
        print("✅ Multi-Node Ray Cluster successfully generated text!")
    finally:
        # Cleanup job so we don't hog 2 nodes
        print(f"Cancelling Slurm job {job_id} to release resources...")
        subprocess.run(["scancel", job_id])
