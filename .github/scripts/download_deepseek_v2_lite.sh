#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Prepare deepseek-ai/DeepSeek-V2-Lite-Chat for CPU CI runs by fetching
# only the small config/tokenizer files from HuggingFace and populating
# the HF hub cache. Model weights are intentionally NOT downloaded --
# the caller is expected to pass ``--load-format dummy`` to vLLM so
# weights are random-initialised, keeping runner RAM/disk low.
#
# Environment:
#   HF_TOKEN                    optional HF token (unauthenticated works too)
#   DEEPSEEK_V2_LITE_MODEL_ID   repo id override (default:
#                               deepseek-ai/DeepSeek-V2-Lite-Chat)
#   DEEPSEEK_V2_LITE_FILES      space-separated allow-pattern list for
#                               huggingface_hub.snapshot_download; default
#                               pulls config + tokenizer + generation only.

set -euo pipefail

MODEL_ID="${DEEPSEEK_V2_LITE_MODEL_ID:-deepseek-ai/DeepSeek-V2-Lite-Chat}"
FILES="${DEEPSEEK_V2_LITE_FILES:-config.json configuration_deepseek.py tokenizer.json tokenizer_config.json generation_config.json}"

echo "==> Prefetching ${MODEL_ID} config/tokenizer files (weights skipped)"
echo "    allow_patterns: ${FILES}"

MODEL_ID="${MODEL_ID}" FILES="${FILES}" HF_TOKEN="${HF_TOKEN:-}" \
  python3 - <<'PY'
import os

from huggingface_hub import snapshot_download

model_id = os.environ["MODEL_ID"]
allow = os.environ["FILES"].split()
token = os.environ.get("HF_TOKEN") or None

path = snapshot_download(
    repo_id=model_id,
    allow_patterns=allow,
    token=token,
)
print("Cached %s at %s" % (model_id, path))
PY

echo "==> Done"
