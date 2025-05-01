#!/bin/bash
set -euxo pipefail

# Create and activate virtual environment
python3 -m venv mmlu_venv
source mmlu_venv/bin/activate

# Ensure HF token exists
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HuggingFace token not set. Export HF_TOKEN in secrets or agent env."
  exit 1
fi

# Log in to HuggingFace
huggingface-cli login --token "$HF_TOKEN"
echo "✅ HuggingFace login complete." > setup-status.txt

# Clone VLLM
if [[ ! -d "vllm" ]]; then
  git clone https://github.com/vllm-project/vllm.git
fi

# Patch api_server.py for app.state.engine_client
patch_file="vllm/vllm/entrypoints/api_server.py"
if ! grep -q "app.state.engine_client" "$patch_file"; then
  sed -i '/assert engine is not None/a \
app.state.engine_client = engine' "$patch_file"
  echo "✅ vLLM patch applied to $patch_file" >> setup-status.txt
else
  echo "ℹ️ Patch already present in $patch_file" >> setup-status.txt
fi

# Install vLLM
pip install -e ./vllm

# Install LMCache
if [[ ! -d "LMCache" ]]; then
  git clone https://github.com/openlm-research/LMCache.git
fi
pip install -e ./LMCache

# Install SGLang (optional: use a fork if needed)
if [[ ! -d "sglang" ]]; then
  git clone https://github.com/sgl-project/sglang.git
fi
pip install -e ./sglang