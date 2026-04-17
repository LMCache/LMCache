#!/usr/bin/env bash
# Per-job environment setup: installs vLLM nightly + LMCache from source.
# Called at the start of every CI job.
set -euo pipefail

# Print the failing command and line number on any error.
trap 'echo "ERROR: setup-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# ── GPU health pre-check ────────────────────────────────────
# Fail fast if GPUs are occupied by stale host processes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80

echo "--- :python: Installing vLLM nightly (pinned to cu130 index)"
# The base image is nvidia/cuda:13.0.2-devel-ubuntu24.04 (system nvcc 13).
# vLLM's generic nightly index (wheels.vllm.ai/nightly/vllm/) non-deterministically
# resolves to either a cu128 or a cu130 torch wheel depending on which wheel
# vLLM's nightly CI happened to publish that day. When the resolver picks a
# cu128 torch, torch.utils.cpp_extension._check_cuda_version aborts the
# LMCache editable install with:
#   RuntimeError: The detected CUDA version (13.0) mismatches the version
#   that was used to compile PyTorch (12.8).
#
# Pin to the cu130 sub-index so torch.version.cuda is always "13.0" and
# matches the base image. This also lets us drop the HTML-scraping + apt
# cuda-compiler alignment dance that lived here before.
# (See https://docs.vllm.ai/ install tips → Nightly → CUDA 13.0.)
uv pip install -U "vllm[runai,tensorizer,flashinfer]" --pre \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --index-strategy unsafe-best-match

# vLLM nightlies periodically add eager imports of packages that aren't in
# their declared deps (e.g. `pandas` from vllm/_aiter_ops.py). Probe-import
# vllm's CLI entry point and auto-install any ModuleNotFoundError modules
# so the job keeps going. Capped to avoid infinite loops; every auto-install
# is logged so the drift is visible in the build output.
MAX_AUTO_INSTALL=5
for i in $(seq 1 "$MAX_AUTO_INSTALL"); do
    if err=$(python -c "from vllm.entrypoints.cli.main import main" 2>&1); then
        break
    fi
    mod=$(printf '%s\n' "$err" | sed -n "s/.*No module named '\([^']*\)'.*/\1/p" | head -1)
    if [[ -z "$mod" ]]; then
        echo "vLLM import failed with a non-ModuleNotFoundError:" >&2
        echo "$err" >&2
        exit 1
    fi
    if [[ "$i" == "$MAX_AUTO_INSTALL" ]]; then
        echo "Hit $MAX_AUTO_INSTALL auto-install retries; last missing module: $mod" >&2
        echo "$err" >&2
        exit 1
    fi
    echo "Auto-installing missing vLLM runtime dep: $mod"
    uv pip install "$mod"
done

echo "--- :mag: Verifying torch CUDA matches system nvcc"
# Sanity check: fail fast with a clear message if the cu130 pin above
# somehow didn't produce a cu13x torch. Previously this mismatch surfaced
# deep inside ninja as a cryptic `cusparse.h: No such file or directory`;
# catching it here makes the failure mode obvious.
python - <<'PY'
import subprocess, sys, torch
tc = torch.version.cuda or ""
try:
    nv = subprocess.check_output(["nvcc", "--version"], text=True)
    sys_major = next(
        (line.split("release ")[1].split(",")[0].split(".")[0]
         for line in nv.splitlines() if "release " in line),
        "",
    )
except Exception:
    sys_major = ""
torch_major = tc.split(".")[0] if tc else ""
print(f"torch.version.cuda={tc!r}; system nvcc major={sys_major!r}")
if torch_major and sys_major and torch_major != sys_major:
    sys.exit(
        f"CUDA major mismatch: torch={torch_major} vs nvcc={sys_major}. "
        "Check the vLLM nightly cu130 index pin in setup-env.sh."
    )
PY

echo "--- :python: Installing LMCache from source"
uv pip install -e . --no-build-isolation

echo "--- :white_check_mark: Environment ready"
python -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation')"
