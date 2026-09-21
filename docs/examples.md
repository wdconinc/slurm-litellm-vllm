# Examples

This section contains executable examples demonstrating how to use the decentralized Slurm+vLLM infrastructure in a real multi-job workload.

## Elements Job Array

The **Elements Job Array** demonstrates the multi-job distributed fleet workflow. It uses a **Slurm Job Array** to launch 20 parallel, minimal-CPU compute jobs. 

Each job:
1. Sources the `endpoint.env` file published by the centralized GPU instance.
2. Passes its `SLURM_ARRAY_TASK_ID` (1 through 20) to a Python script (`query_element.py`).
3. The Python script connects to the decentralized LiteLLM proxy and asks the LLM to write a document about the chemical element matching that atomic number.
4. Outputs the response to a text file (e.g., `element_1.txt` for Hydrogen).

### How to Run

1. **Start the LLM Backend**:
   Ensure you have a running GPU backend first by executing the unified start script from the root of the repository:
   ```bash
   ./bin/start.sh leanstral
   ```

2. **Wait for Initialization**:
   Wait until `start.sh` confirms the server is healthy and `run/endpoint.env` is created.

3. **Submit the Array**:
   Submit the job array:
   ```bash
   sbatch examples/elements_job_array.sh
   ```

4. **View Results**:
   As the workers finish, you will see output files generated in your directory (`element_1.txt`, `element_2.txt`, etc.).

---

### Source Code

#### `examples/elements_job_array.sh`
```bash
#!/bin/bash
#SBATCH --job-name=elements_array
#SBATCH --array=1-20          # Launch 20 parallel jobs (atomic numbers 1 to 20)
#SBATCH --cpus-per-task=1     # Minimal CPU requirement
#SBATCH --mem=1G              # Minimal Memory requirement
#SBATCH --time=00:10:00       # Short duration
#SBATCH --output=element_job_%A_%a.log

# 1. Load the centralized LLM endpoint published by bin/start.sh
ENDPOINT_FILE="/home/wdconinc/git/slurm-litellm-vllm/run/endpoint.env"

if [ ! -f "$ENDPOINT_FILE" ]; then
    echo "ERROR: Endpoint file $ENDPOINT_FILE not found."
    echo "Please ensure you have launched the vLLM proxy via bin/start.sh first."
    exit 1
fi

# This populates OPENAI_API_BASE, OPENAI_API_KEY, and OPENAI_MODEL
source "$ENDPOINT_FILE"

# 2. Run the Python script to query the LLM
# The Python openai library automatically detects the environment variables
echo "Worker ${SLURM_ARRAY_TASK_ID} starting..."

python examples/query_element.py ${SLURM_ARRAY_TASK_ID}

echo "Worker ${SLURM_ARRAY_TASK_ID} finished."
```

#### `examples/query_element.py`
```python
import os
import sys
from openai import OpenAI

def main():
    if len(sys.argv) < 2:
        print("Usage: python query_element.py <atomic_number>")
        sys.exit(1)
        
    atomic_number = sys.argv[1]
    
    # The OpenAI client automatically inherits OPENAI_API_BASE and OPENAI_API_KEY
    # from the environment variables sourced from endpoint.env
    client = OpenAI()
    
    # We also pull the model name from the environment, defaulting to my-local-model
    model_name = os.getenv("OPENAI_MODEL", "my-local-model")

    prompt = f"Write a brief, factual document about the chemical element with atomic number {atomic_number}. Include its name, symbol, discovery, and primary real-world uses."

    print(f"Querying model '{model_name}' for atomic number {atomic_number}...")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300
    )
    
    result = response.choices[0].message.content
    
    # Save the result to a text file
    output_filename = f"element_{atomic_number}.txt"
    with open(output_filename, "w") as f:
        f.write(result)
        
    print(f"Successfully wrote {output_filename}")

if __name__ == "__main__":
    main()
```
