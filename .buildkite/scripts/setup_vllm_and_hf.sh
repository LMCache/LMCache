#!/bin/bash
set -euxo pipefail

# Make sure HF token is set (or fail fast)
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HuggingFace token not set. Export HF_TOKEN in secrets or agent env."
  exit 1
fi

# Login to HuggingFace
huggingface-cli login --token "$HF_TOKEN"

echo "✅ HuggingFace login complete." > vllm-patch-status.txt

# Patch vllm/entrypoints/api_server.py if needed
patch_file="vllm/vllm/entrypoints/api_server.py"
grep -q "app.state.engine_client" $patch_file || \
  sed -i '/assert engine is not None/a \
app.state.engine_client = engine' $patch_file

echo "✅ vLLM patch applied to $patch_file" >> vllm-patch-status.txt
