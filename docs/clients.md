# Client Configuration

Once your vLLM job is running and the LiteLLM proxy is active on the login node, you must set up an SSH port forward from your local machine:

```bash
ssh -L 4000:127.0.0.1:4000 your_username@grex.hpc.umanitoba.ca
```

With the port forwarded, your local machine can communicate directly with the LiteLLM proxy as if it were a local OpenAI-compatible endpoint.

Because LiteLLM provides a strict OpenAI-compatible API format, almost any modern LLM client can be configured to use your HPC infrastructure by simply overriding the **Base URL** and the **API Key**.

Below are configuration guides for common clients.

---

## OpenCode

[OpenCode](https://github.com/) can be configured to use your local cluster by providing a custom `opencode.json` configuration file. Create or update `opencode.json` in the root of your project:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "slurm-cluster": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "HPC vLLM Cluster",
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "sk-hpc-secret-key"
      },
      "models": {
        "qwen": {
          "name": "Qwen3 Coder Next"
        },
        "leanstral": {
          "name": "Leanstral (119B)"
        },
        "llama-70b": {
          "name": "Llama 3 70B Instruct"
        }
      }
    }
  }
}
```

Then you can specify the provider and model when running OpenCode, e.g., `opencode --model slurm-cluster:leanstral`.

---

## Continue (VS Code & JetBrains)

[Continue](https://continue.dev/) is a popular open-source AI code assistant for your IDE. To use your HPC model, add the following entry to the `models` array in your `config.json` (accessible via the Continue gear icon):

```json
{
  "models": [
    {
      "title": "HPC vLLM Cluster",
      "provider": "openai",
      "model": "my-local-model",
      "apiBase": "http://localhost:4000/v1",
      "apiKey": "sk-hpc-secret-key"
    }
  ]
}
```

---

## OpenAI Python SDK

If you are writing your own Python scripts, agents, or evaluation pipelines, use the official OpenAI SDK and override the `base_url`.

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:4000/v1",
    api_key="sk-hpc-secret-key"
)

response = client.chat.completions.create(
    model="my-local-model",
    messages=[{"role": "user", "content": "How do I reverse a string in Python?"}]
)

print(response.choices[0].message.content)
```

---

## LangChain (Python)

If your application uses LangChain, use the `ChatOpenAI` class and pass the overridden parameters:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="my-local-model",
    openai_api_base="http://localhost:4000/v1",
    openai_api_key="sk-hpc-secret-key",
    max_tokens=512
)

response = llm.invoke("Explain HPC scheduling.")
print(response.content)
```

---

## Curl (Command Line)

For quick health checks or shell scripts, you can make raw HTTP requests using `curl`:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-hpc-secret-key" \
  -d '{
    "model": "my-local-model",
    "messages": [
      {
        "role": "user",
        "content": "Hello, are you receiving my prompt?"
      }
    ]
  }'
```
