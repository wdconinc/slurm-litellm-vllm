# Python Migration Roadmap

As the Slurm + vLLM infrastructure scales to support larger multi-node Ray deployments, complex observability (Langfuse), and idle-timeout watchdogs, the purely Bash-based orchestration scripts (`bin/start.sh` and `bin/vllm.sh`) will approach the limits of their maintainability.

This document outlines the architectural roadmap for transitioning from a "Pure Bash" architecture to a **"Thin Bash, Fat Python"** architecture.

## The Vision: Thin Bash, Fat Python
Slurm is fundamentally designed around Bash scripts containing `#SBATCH` directives. We cannot eliminate Bash entirely. Instead, Bash will be relegated strictly to environment bootstrapping, while Python will handle all complex state management, process orchestration, and configuration generation.

---

## Phase 1: The Compute Node Orchestrator
**Goal**: Replace the complex 250+ line `bin/vllm.sh` with a modular Python daemon.

### Current State
`bin/vllm.sh` handles Singularity execution, Ray cluster initialization, dynamic YAML generation, background process trapping, and an idle timeout watchdog using `curl` and `awk`.

### Proposed Architecture
1. **`bin/vllm.sh` (Shrunk to ~15 lines)**
   - Contains `#SBATCH` headers.
   - Loads the Singularity module.
   - Activates `.venv`.
   - Executes `python -m src.orchestrator`.
2. **`src/orchestrator/` (New Python Module)**
   - **`config.py`**: Uses `pydantic` to parse environment variables (`.env`), validate Hugging Face tokens, and dynamically dump the LiteLLM config to YAML.
   - **`ray_manager.py`**: Uses `subprocess.Popen` to initialize the Ray head and worker nodes, tracking their PIDs.
   - **`server.py`**: Boots the vLLM Singularity container and waits for the `/health` endpoint to return HTTP 200.
   - **`watchdog.py`**: A threaded daemon that polls `http://127.0.0.1:8000/metrics`, parses the exact JSON/Prometheus payload to track active requests, and gracefully sends `SIGTERM` to the container when the idle threshold is reached.

---

## Phase 2: The Login Node Submitter
**Goal**: Replace `bin/start.sh` with a robust Python CLI.

### Current State
`bin/start.sh` submits the job, uses regex to parse the Job ID from the terminal output, and loops over `squeue` using `echo -ne \r` to display progress.

### Proposed Architecture
1. **`bin/start.py` (CLI entrypoint)**
   - Built with `typer` or `click` to provide a robust CLI (e.g., `python bin/start.py --model mistral --nodes 2 --timeout 30`).
2. **`SlurmManager` Class**
   - Integrates with Meta's `submitit` library (or uses structured `subprocess` calls) to submit jobs.
   - Uses `squeue --json` to fetch structured state data (e.g., waiting reason, queue position, estimated start time).
   - Provides a rich, terminal-based UI (using `rich`) showing a progress bar of the deployment status.

---

## Phase 3: Advanced Fleet Capabilities
Once Python is orchestrating the infrastructure, advanced capabilities become trivial to implement:

- **Autoscaling Router**: Instead of a 1:1 mapping between LiteLLM and vLLM, the Python orchestrator on the login node can monitor queue depth. If Lean4 workers queue up too many requests, the Python script automatically calls `sbatch` to spin up a *second* vLLM job, dynamically adding it to the LiteLLM router config.
- **Strict Teardowns**: Python's `atexit` handlers and explicit signal catching guarantee that `endpoint.env` is safely deleted and Singularity zombie processes are fully reaped when Slurm preempts a job.
- **Programmatic API**: The orchestration logic can be imported into Python notebooks (`import slurm_llm`), allowing researchers to boot LLM backends directly from Jupyter without touching the terminal.
