#!/usr/bin/env bash
# Per-job environment setup: installs vLLM nightly + LMCache from source.
# Called at the start of every CI job.
set -euo pipefail

echo "--- :python: Installing vLLM nightly"
uv pip install --prerelease=allow \
    'vllm[runai,tensorizer,flashinfer]' \
    --extra-index-url https://wheels.vllm.ai/nightly \
    --index-strategy unsafe-best-match

echo "--- :python: Installing LMCache from source"
uv pip install -e . --no-build-isolation

echo "--- :white_check_mark: Environment ready"
python -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation')"
