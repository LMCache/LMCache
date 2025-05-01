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

patch_file="vllm/vllm/entrypoints/api_server.py"

# 1. Patch inside init_app (usually already there, but idempotent)
if ! grep -q "app.state.engine_client = engine" "$patch_file"; then
  sed -i '/assert engine is not None/a \
app.state.engine_client = engine' "$patch_file"
  echo "✅ Patch 1 applied to init_app" >> setup-status.txt
else
  echo "ℹ️ Patch 1 already present in init_app" >> setup-status.txt
fi

# 2. Patch inside run_server (new location)
# Match: assert engine is not None
# Insert: app.state.engine_client = engine
if ! grep -A1 "await init_app" "$patch_file" | grep -q "app.state.engine_client"; then
  sed -i '/assert engine is not None/a \
app.state.engine_client = engine' "$patch_file"
  echo "✅ Patch 2 applied to run_server" >> setup-status.txt
else
  echo "ℹ️ Patch 2 already present in run_server" >> setup-status.txt
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