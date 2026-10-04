# Supported Models

The following models are currently pre-configured and validated for the cluster.

| Key | Fullname | Context Window | Partition | GPUs | CPUs | RAM | Time |
|---|---|---|---|---|---|---|---|
| `qwen` | `Qwen/Qwen3-Coder-Next` | 128K | `lgpu` | 2 | 64 | 128G | 04:00:00 |
| `smollm-cpu` | `HuggingFaceTB/SmolLM-135M-Instruct` | 2K | `skylake` | 0 | 2 | 16G | 01:00:00 |
| `smollm-cpu-ray` | `HuggingFaceTB/SmolLM-135M-Instruct` | 2K | `skylake` | 0 | 2 | 16G | 01:00:00 |
| `llama-70b` | `meta-llama/Llama-3.1-70B-Instruct` | 32K | `lgpu` | 2 | 32 | 128G | 04:00:00 |
| `mistral-large` | `neuralmagic/Mistral-Large-Instruct-2407-FP8` | 32K | `lgpu` | 2 | 32 | 128G | 04:00:00 |
| `leanstral-2603` | `mistralai/Leanstral-2603` | 64K | `lgpu` | 2 | 32 | 256G | 04:00:00 |
| `leanstral-1.5` | `mistralai/Leanstral-1.5-119B-A6B` | 64K | `lgpu` | 2 | 32 | 256G | 04:00:00 |
| `llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 32K | `agpu` | 2 | 24 | 128G | 04:00:00 |
| `qwen-14b` | `Qwen/Qwen2.5-14B-Instruct` | 128K | `agpu` | 2 | 24 | 128G | 04:00:00 |
| `qwen-32b` | `Qwen/Qwen2.5-32B-Instruct` | 128K | `agpu` | 2 | 24 | 128G | 04:00:00 |

## Detailed Configurations

### `qwen` (Aliases: `qwen3`)
- **HuggingFace Path:** `Qwen/Qwen3-Coder-Next`
- **Context Window:** 128K
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

### `smollm-cpu`
- **HuggingFace Path:** `HuggingFaceTB/SmolLM-135M-Instruct`
- **Context Window:** 2K
- **Hardware Request:** 1 nodes, 0 GPUs, 2 CPUs, 16G RAM on `skylake`
- **vLLM Arguments:**
  - `--gpu-memory-utilization 0.1`
  - `--max-model-len 2048`
  - `--enforce-eager`

### `smollm-cpu-ray`
- **HuggingFace Path:** `HuggingFaceTB/SmolLM-135M-Instruct`
- **Context Window:** 2K
- **Hardware Request:** 2 nodes, 0 GPUs, 2 CPUs, 16G RAM on `skylake`
- **vLLM Arguments:**
  - `--gpu-memory-utilization 0.1`
  - `--max-model-len 2048`
  - `--enforce-eager`

### `llama-70b`
- **HuggingFace Path:** `meta-llama/Llama-3.1-70B-Instruct`
- **Context Window:** 32K
- **Hardware Request:** 2 nodes, 2 GPUs, 32 CPUs, 128G RAM on `lgpu`
- **vLLM Arguments:**
  - `--tensor-parallel-size 2`
  - `--pipeline-parallel-size 2`
  - `--max-model-len 32768`
  - `--gpu-memory-utilization 0.95`

### `mistral-large`
- **HuggingFace Path:** `neuralmagic/Mistral-Large-Instruct-2407-FP8`
- **Context Window:** 32K
- **Hardware Request:** 2 nodes, 2 GPUs, 32 CPUs, 128G RAM on `lgpu`
- **vLLM Arguments:**
  - `--tensor-parallel-size 2`
  - `--pipeline-parallel-size 2`
  - `--max-model-len 32768`
  - `--gpu-memory-utilization 0.95`

### `leanstral-2603`
- **HuggingFace Path:** `mistralai/Leanstral-2603`
- **Context Window:** 64K
- **Hardware Request:** 2 nodes, 2 GPUs, 32 CPUs, 256G RAM on `lgpu`
- **vLLM Arguments:**
  - `--enable-auto-tool-choice`
  - `--tool-call-parser mistral`
  - `--tensor-parallel-size 4`
  - `--quantization fp8`
  - `--max-model-len 65536`
  - `--gpu-memory-utilization 0.90`
  - `--trust-remote-code`

### `leanstral-1.5`
- **HuggingFace Path:** `mistralai/Leanstral-1.5-119B-A6B`
- **Context Window:** 64K
- **Hardware Request:** 2 nodes, 2 GPUs, 32 CPUs, 256G RAM on `lgpu`
- **vLLM Arguments:**
  - `--enable-auto-tool-choice`
  - `--tool-call-parser mistral`
  - `--tensor-parallel-size 4`
  - `--quantization fp8`
  - `--max-model-len 65536`
  - `--gpu-memory-utilization 0.90`
  - `--trust-remote-code`

### `llama-8b`
- **HuggingFace Path:** `meta-llama/Llama-3.1-8B-Instruct`
- **Context Window:** 32K
- **Hardware Request:** 1 nodes, 2 GPUs, 24 CPUs, 128G RAM on `agpu`
- **vLLM Arguments:**
  - `--tensor-parallel-size 2`
  - `--max-model-len 32768`
  - `--gpu-memory-utilization 0.90`

### `qwen-14b`
- **HuggingFace Path:** `Qwen/Qwen2.5-14B-Instruct`
- **Context Window:** 128K
- **Hardware Request:** 1 nodes, 2 GPUs, 24 CPUs, 128G RAM on `agpu`
- **vLLM Arguments:**
  - `--tensor-parallel-size 2`
  - `--max-model-len 131072`
  - `--gpu-memory-utilization 0.90`
  - `--trust-remote-code`
  - `--enable-auto-tool-choice`
  - `--tool-call-parser hermes`

### `qwen-32b`
- **HuggingFace Path:** `Qwen/Qwen2.5-32B-Instruct`
- **Context Window:** 128K
- **Hardware Request:** 2 nodes, 2 GPUs, 24 CPUs, 128G RAM on `agpu`
- **vLLM Arguments:**
  - `--tensor-parallel-size 2`
  - `--pipeline-parallel-size 2`
  - `--max-model-len 131072`
  - `--gpu-memory-utilization 0.90`
  - `--trust-remote-code`
  - `--enable-auto-tool-choice`
  - `--tool-call-parser hermes`

