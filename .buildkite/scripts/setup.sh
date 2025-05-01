#!/bin/bash
set -euxo pipefail

source /dataheart/yihua98/Applications/anaconda3/envs/buildkite/bin/activate

# Ensure HF token exists
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "[ERROR] HuggingFace token not set. Export HF_TOKEN in secrets or agent env."
  exit 1
fi

pip install vllm
pip install torch==2.6.0
pip install lmcache