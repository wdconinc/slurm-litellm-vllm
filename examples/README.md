# Examples

This directory contains executable examples demonstrating how to use the decentralized Slurm+vLLM infrastructure.

## Elements Job Array

The `elements_job_array.sh` script demonstrates the multi-job distributed fleet workflow. It uses a **Slurm Job Array** to launch 20 parallel, minimal-CPU compute jobs. 

Each job:
1. Sources the `endpoint.env` file published by the centralized GPU instance.
2. Passes its `SLURM_ARRAY_TASK_ID` to a python script (`query_element.py`).
3. The python script connects to the LiteLLM proxy and asks the LLM to write a document about the chemical element matching that atomic number.
4. Outputs the response to a text file (e.g., `element_1.txt` for Hydrogen).

### How to Run

1. **Start the LLM Backend**:
   Ensure you have a running GPU backend first by executing the unified start script from the root of the repository:
   ```bash
   ./bin/start.sh mistral
   ```

2. **Wait for Initialization**:
   Wait until `start.sh` confirms the server is healthy and `~/litellm/etc/endpoint.env` is created.

3. **Submit the Array**:
   Submit the job array:
   ```bash
   sbatch examples/elements_job_array.sh
   ```

4. **View Results**:
   As the workers finish, you will see output files generated in your directory (`element_1.txt`, `element_2.txt`, etc.).
