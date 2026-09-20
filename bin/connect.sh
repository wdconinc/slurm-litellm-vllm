#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <SSH_HOST>"
    echo "Example: $0 username@grex.hpc.umanitoba.ca"
    exit 1
fi

HOST=$1

echo "Fetching proxy endpoint details from $HOST..."

# Assuming the slurm-litellm-vllm directory is in the home folder or we find it
ENDPOINT_BASE=$(ssh -q "$HOST" "cat slurm-litellm-vllm/run/endpoint.env 2>/dev/null | grep '^OPENAI_API_BASE='")

if [ -z "$ENDPOINT_BASE" ]; then
    echo "Error: Could not read OPENAI_API_BASE from slurm-litellm-vllm/run/endpoint.env on $HOST."
    echo "Ensure that the remote cluster has been started and the endpoint.env file exists."
    exit 1
fi

IP=$(echo "$ENDPOINT_BASE" | sed -n 's|.*://\([^:/]*\).*|\1|p')

if [ -z "$IP" ]; then
    echo "Error: Could not parse IP address from $ENDPOINT_BASE"
    exit 1
fi

echo "Found internal IP: $IP"
echo "Establishing local port forward (localhost:4000 -> $IP:4000)..."

ssh -L 4000:"$IP":4000 "$HOST"
