# Supported Models

The following models are currently pre-configured and validated for the cluster.

| Key | Fullname | Partition | GPUs | CPUs | RAM | Time |
|---|---|---|---|---|---|---|
| `qwen` | `Qwen/Qwen3-Coder-Next` | `lgpu` | 2 | 64 | 128G | 04:00:00 |
| `mistral` | `sahilchachra/Leanstral-1.5-119B-A6B-NVFP4` | `lgpu` | 2 | 64 | 128G | 04:00:00 |
| `smollm-cpu` | `HuggingFaceTB/SmolLM-135M-Instruct` | `skylake` | 0 | 2 | 16G | 01:00:00 |
| `smollm-cpu-ray` | `HuggingFaceTB/SmolLM-135M-Instruct` | `skylake` | 0 | 2 | 16G | 01:00:00 |

## Detailed Configurations

### `qwen` (Aliases: `qwen3`)
- **HuggingFace Path:** `Qwen/Qwen3-Coder-Next`
- **Hardware Request:** 1 nodes, 2 GPUs, 64 CPUs, 128G RAM on `lgpu`
- **vLLM Arguments:**
  - `--enable-auto-tool-choice`
  - `--tool-call-parser qwen3_xml`
  - `--max-model-len 131072`
  - `--gpu-memory-utilization 0.95`
  - `--quantization fp8`
  - `--enable-prefix-caching`
  - `--enable-chunked-prefill`
  - `--trust-remote-code`

### `mistral` (Aliases: `leanstral`)
- **HuggingFace Path:** `sahilchachra/Leanstral-1.5-119B-A6B-NVFP4`
- **Hardware Request:** 1 nodes, 2 GPUs, 64 CPUs, 128G RAM on `lgpu`
- **vLLM Arguments:**
  - `--limit-mm-per-prompt image=0`
  - `--quantization compressed-tensors`
  - `--max-model-len 32764`
  - `--gpu-memory-utilization 0.90`
  - `--tokenizer-mode mistral`
  - `--tool-call-parser mistral`
  - `--reasoning-parser mistral`

### `smollm-cpu`
- **HuggingFace Path:** `HuggingFaceTB/SmolLM-135M-Instruct`
- **Hardware Request:** 1 nodes, 0 GPUs, 2 CPUs, 16G RAM on `skylake`
- **vLLM Arguments:**
  - `--gpu-memory-utilization 0.1`
  - `--max-model-len 2048`
  - `--enforce-eager`

### `smollm-cpu-ray`
- **HuggingFace Path:** `HuggingFaceTB/SmolLM-135M-Instruct`
- **Hardware Request:** 2 nodes, 0 GPUs, 2 CPUs, 16G RAM on `skylake`
- **vLLM Arguments:**
  - `--gpu-memory-utilization 0.1`
  - `--max-model-len 2048`
  - `--enforce-eager`

