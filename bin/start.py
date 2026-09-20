import os
import sys
import time
import subprocess


from typing import Optional, Any


def deep_merge(dict1: dict, dict2: dict) -> dict:
    for k, v in dict2.items():
        if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
            deep_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def submit_job(model_key: str) -> Optional[str]:
    import yaml

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml")
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading config/models.yaml: {e}")
        sys.exit(1)

    if model_key in data and isinstance(data[model_key], str):
        model_key = data[model_key]  # resolve alias

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
        print(
            "Warning: Could not parse job ID from sbatch output. Will wait for config file indefinitely."
        )
    else:
        print(
            f"Waiting for Slurm job {job_id} to initialize and generate LiteLLM config..."
        )
        print(
            "(This may take a few minutes while the container starts and the model loads)"
        )

    return job_id


def monitor_job(job_id: Optional[str]) -> str:
    endpoint_file = os.path.join(os.path.dirname(__file__), "..", "run", "endpoint.env")
    if os.path.exists(endpoint_file):
        os.remove(endpoint_file)

    start_time = time.time()

    while not os.path.exists(endpoint_file):
        elapsed = int(time.time() - start_time)
        if job_id:
            squeue_res = subprocess.run(
                ["squeue", "-h", "-j", job_id, "-o", "%T"],
                capture_output=True,
                text=True,
            )
            status = squeue_res.stdout.strip()

            if not status:
                print(
                    f"\n❌ Error: Slurm job {job_id} is no longer in the queue. It likely failed."
                )
                print(f"Please check vllm_{job_id}.log for details.")
                sys.exit(1)

            sys.stdout.write(
                f"\r⏳ Waiting for endpoint.env... Job {job_id} is [ {status} ] (Elapsed: {elapsed}s)   "
            )
            sys.stdout.flush()
        else:
            sys.stdout.write(
                f"\r⏳ Waiting for endpoint.env... (Elapsed: {elapsed}s)   "
            )
            sys.stdout.flush()

        time.sleep(5)

    print("\n")
    return endpoint_file


def print_success(endpoint_file: str) -> str:
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

    abs_endpoint = os.path.abspath(endpoint_file)
    user = os.getenv("USER", "$USER")

    print("==========================================================")
    print("✅ Success! Decentralized LLM Proxy is running.")
    print("==========================================================")
    print("For Cluster Workers / Background Jobs:")
    print("  Before running your worker scripts, load the endpoint:")
    print(f"  source {abs_endpoint}")
    print("\nFor Local Laptop Access (SSH Port Forwarding):")
    print("  Run this command on your laptop to tunnel to the compute node:")
    print(f"  ssh -L 4000:{compute_ip}:4000 {user}@grex.hpc.umanitoba.ca")
    print("\n  Or add this snippet to your ~/.ssh/config:")
    print("  Host proxy")
    print("      HostName grex.hpc.umanitoba.ca")
    print(f"      User {user}")
    print(f"      LocalForward 4000 {compute_ip}:4000")
    print("\n  Test your connection (from another terminal on your laptop):")
    print("  curl http://localhost:4000/v1/chat/completions \\")
    print('    -H "Content-Type: application/json" \\')
    print('    -H "Authorization: Bearer sk-hpc-secret-key" \\')
    print("    -d '{")
    print('      "model": "my-local-model",')
    print('      "messages": [{"role": "user", "content": "Hello!"}]')
    print("    }'")
    print("==========================================================")
    return compute_ip


def watch_status(job_id: Optional[str], compute_ip: str) -> None:
    import urllib.request
    import yaml
    import signal

    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml")
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
            defaults = data.get("defaults", {})
            max_idle_mins = int(defaults.get("watchdog", {}).get("max_idle_mins", 15))
    except Exception:
        max_idle_mins = 15

    idle_mins = 0

    def handle_sigtstp(signum: int, frame: Any) -> None:
        print(
            "\n\n[Monitor] Monitor paused. The vLLM server and watchdog continue to run independently on the cluster!"
        )
        print(
            "          Type 'fg' to resume monitoring, or 'bg' to monitor silently in the background."
        )
        sys.stdout.flush()
        signal.signal(signal.SIGTSTP, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTSTP)

    def handle_sigcont(signum: int, frame: Any) -> None:
        signal.signal(signal.SIGTSTP, handle_sigtstp)
        time_left = "Unknown"
        if job_id:
            try:
                squeue_res = subprocess.run(
                    ["squeue", "-h", "-j", job_id, "-o", "%L"],
                    capture_output=True,
                    text=True,
                )
                if squeue_res.stdout.strip():
                    time_left = squeue_res.stdout.strip()
            except Exception:
                pass

        mins_to_watchdog = max_idle_mins - idle_mins
        print(
            f"\n[Monitor] Resumed. Job Time Left: {time_left} | Watchdog kills in: {mins_to_watchdog} mins"
        )
        sys.stdout.flush()

    signal.signal(signal.SIGTSTP, handle_sigtstp)
    signal.signal(signal.SIGCONT, handle_sigcont)

    print(
        "\n[Monitor] Watching active session (Ctrl+C to stop monitor without killing job)..."
    )

    try:
        while True:
            time_left = "Unknown"
            if job_id:
                squeue_res = subprocess.run(
                    ["squeue", "-h", "-j", job_id, "-o", "%L"],
                    capture_output=True,
                    text=True,
                )
                time_left = squeue_res.stdout.strip()
                if not time_left:
                    print(f"\n[Monitor] Job {job_id} is no longer running.")
                    break

            active_reqs = 0
            metrics_available = False
            try:
                req = urllib.request.Request(f"http://{compute_ip}:8000/metrics")
                with urllib.request.urlopen(req, timeout=5) as response:
                    metrics = response.read().decode("utf-8")
                    metrics_available = True
                for line in metrics.splitlines():
                    if line.startswith("vllm:num_requests_") and not line.startswith(
                        "#"
                    ):
                        active_reqs += float(line.split()[1])

                if active_reqs == 0:
                    idle_mins += 1
                else:
                    idle_mins = 0
            except Exception:
                idle_mins = 0

            mins_to_watchdog = max_idle_mins - idle_mins
            is_fg = True
            try:
                if sys.stdout.isatty():
                    is_fg = os.getpgrp() == os.tcgetpgrp(sys.stdout.fileno())
            except Exception:
                pass

            if is_fg:
                if metrics_available:
                    sys.stdout.write(
                        f"\r⏱️  Job Time Left: {time_left} | 🐕 Watchdog kills in: {mins_to_watchdog} mins (Active Reqs: {int(active_reqs)})   "
                    )
                else:
                    sys.stdout.write(
                        f"\r⏱️  Job Time Left: {time_left} | 🐕 Watchdog: {max_idle_mins} mins (Metrics unreachable)   "
                    )

                sys.stdout.flush()

            if metrics_available and mins_to_watchdog <= 0:
                print("\n[Monitor] Watchdog limit reached on compute node. Exiting.")
                break

            time.sleep(60)
    except KeyboardInterrupt:
        print("\n[Monitor] Stopped monitoring. Job is still running on cluster.")


def main() -> None:
    model_key = sys.argv[1] if len(sys.argv) > 1 else "mistral"
    print(f"Submitting vLLM Slurm job for model: {model_key}...")
    job_id = submit_job(model_key)
    endpoint_file = monitor_job(job_id)
    compute_ip = print_success(endpoint_file)
    watch_status(job_id, compute_ip)


if __name__ == "__main__":
    main()
