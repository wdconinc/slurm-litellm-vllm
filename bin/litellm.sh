# Optional but recommended: set a master key to keep other login node users out
export LITELLM_MASTER_KEY="sk-hpc-secret-key" 
export OPENAI_API_KEY="not-needed"

~/litellm/bin/litellm --config ~/litellm/etc/dynamic_litellm_config.yaml --host 127.0.0.1 --port 4000
