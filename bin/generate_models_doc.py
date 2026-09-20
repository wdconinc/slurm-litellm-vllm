import yaml
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_path = os.path.join(repo_root, "config", "models.yaml")
docs_path = os.path.join(repo_root, "docs", "models.md")

with open(config_path, "r") as f:
    data = yaml.safe_load(f)

models = {k: v for k, v in data.items() if k != "defaults" and isinstance(v, dict)}
aliases = {k: v for k, v in data.items() if isinstance(v, str)}

with open(docs_path, "w") as f:
    f.write("# Supported Models\n\n")
    f.write("The following models are currently pre-configured and validated for the cluster.\n\n")
    f.write("| Key | Fullname | Partition | GPUs | CPUs | RAM | Time |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    
    for key, model in models.items():
        slurm = model.get("slurm", {})
        f.write(f"| `{key}` | `{model.get('fullname', '')}` | `{slurm.get('partition', '')}` | {slurm.get('gpus_per_node', 0)} | {slurm.get('cpus_per_task', 1)} | {slurm.get('mem', '')} | {slurm.get('time', '')} |\n")
        
    f.write("\n## Detailed Configurations\n\n")
    
    for key, model in models.items():
        model_aliases = [k for k, v in aliases.items() if v == key]
        alias_str = f" (Aliases: `{', '.join(model_aliases)}`)" if model_aliases else ""
        f.write(f"### `{key}`{alias_str}\n")
        f.write(f"- **HuggingFace Path:** `{model.get('fullname', '')}`\n")
        slurm = model.get("slurm", {})
        f.write(f"- **Hardware Request:** {slurm.get('nodes', 1)} nodes, {slurm.get('gpus_per_node', 0)} GPUs, {slurm.get('cpus_per_task', 1)} CPUs, {slurm.get('mem', '')} RAM on `{slurm.get('partition', '')}`\n")
        args = model.get("vllm", {}).get("args", [])
        f.write(f"- **vLLM Arguments:**\n")
        for arg in args:
            f.write(f"  - `{arg}`\n")
        f.write("\n")

print(f"Generated {docs_path}")
