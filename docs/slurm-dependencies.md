# Slurm Job Dependencies

You can entirely automate your AI pipelines by submitting the LLM backend and your worker scripts as a chain of dependent Slurm jobs. 

Without needing to make any modifications to the codebase, the architecture natively supports the constraint of the LLM job **starting first** and **ending last**.

## The Pattern

### 1. The Submission Script
Instead of using the interactive `bin/start.sh` monitor, you submit the LLM directly to the queue and capture its Job ID using the `--parsable` flag. You then submit your worker jobs and instruct Slurm to wait until the LLM job has begun executing (`after:<job_id>`).

```bash
#!/bin/bash

# 1. Submit the LLM Backend
# We use --parsable to extract just the Job ID
LLM_JOB_ID=$(sbatch --parsable bin/vllm.sh mistral-large)
echo "Submitted LLM Backend (Job ID: $LLM_JOB_ID)"

# 2. Submit the Worker Job array
# --dependency=after ensures the workers stay queued until the LLM node is allocated
sbatch --dependency=after:${LLM_JOB_ID} --array=1-10 my_worker_job.sh
echo "Submitted Worker array, waiting on LLM backend."
```

### 2. The Worker Script (`my_worker_job.sh`)
Even though the LLM job has started, `vLLM` takes a few minutes to boot. Your worker script needs to wait until the LLM publishes its IP address and API key to the `run/endpoint.env` file.

```bash
#!/bin/bash
#SBATCH --job-name=llm-worker
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

echo "Worker started. Waiting for LLM Backend to become healthy..."

# Block until the Orchestrator publishes the connection variables
while [ ! -f "run/endpoint.env" ]; do
    sleep 10
done

# Source OPENAI_BASE_URL and OPENAI_API_KEY into the environment
source run/endpoint.env

# Execute your AI processing script!
# The standard OpenAI Python client will automatically pick up the variables.
python examples/query_element.py
```

## How does the LLM Job "End Last"?
The LLM job natively fulfills the requirement of shutting down last thanks to the **Idle Watchdog** built into `src/orchestrator.py`. 

1. As your dependent worker jobs begin generating requests, the watchdog detects the active traffic.
2. When the last worker job finishes and exits, the LLM stops receiving requests.
3. The watchdog timer (default: 15 minutes) begins counting down.
4. Once the timer expires, the orchestrator cleanly shuts down vLLM and terminates the Slurm job to release the GPUs back to the cluster.
