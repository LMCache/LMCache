#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install LMCache + vLLM in CPU-only mode for the cpu_hello_world demo.
# No GPU, CUDA, or nvcc required.
#
# This is a thin wrapper over the two scripts LMCache's own CPU CI runs:
#
#   .github/scripts/install_vllm_cpu.sh     vLLM CPU wheel + the load-bearing
#                                           `vllm-<ver>+cpu` dist-info alias
#   .github/scripts/install_lmcache_cpu.sh  LMCache built with NO_GPU_EXT=1
#                                           (drops the CUDA kernels, keeps the
#                                           common C++ extensions) + the gRPC
#                                           transport bindings
#
# Delegating rather than reimplementing is deliberate. Those scripts carry
# fixes this demo would otherwise have to rediscover one breakage at a time --
# the dist-info alias, an `apache-tvm-ffi<0.1.13` cap that stops xgrammar
# segfaulting on macOS arm64, and the generated gRPC bindings. Read them for
# the full reasoning; each step is commented in place.
#
# What this wrapper adds is the part CI gets from its workflow file rather
# than from those scripts: the system-library preflight and the virtualenv
# check.
#
# Usage (run from anywhere inside the LMCache repo, inside an activated venv):
#   examples/cpu_hello_world/install_cpu.sh
#   PIP_BIN="uv pip" examples/cpu_hello_world/install_cpu.sh
#
# Environment (all optional):
#   PIP_BIN                 pip front-end (default: "pip"; e.g. "uv pip")
#   VLLM_CPU_NIGHTLY_SPEC   vLLM CPU package spec (default: "vllm-cpu-nightly").
#                           Pin a known-good build for reproducibility, e.g.
#                           VLLM_CPU_NIGHTLY_SPEC="vllm-cpu-nightly==<version>".
#                           CI pins per-OS; see the versions it has verified at
#                           https://github.com/LMCache/LMCache/tree/github_nightly_tested_vllm
#   PIP_INSTALL_EXTRA_ARGS  extra flags forwarded to the vLLM install

set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
CI_SCRIPTS="${REPO_ROOT}/.github/scripts"

echo "==> Repo root: ${REPO_ROOT}"
echo "==> Python:    $(python3 --version 2>&1 || true)"
echo "==> pip front-end: ${PIP_BIN:-pip}"

# ------------------------------------------------------------------ #
# Preflight 1: activated virtualenv
# ------------------------------------------------------------------ #
# Installing editable packages into the system Python is a bad idea, and the
# CI scripts invoke bare `python`, which on Ubuntu 20.04+ only resolves
# inside an activated venv.
if ! python3 -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)' 2>/dev/null; then
  cat >&2 <<'MSG'
!! No activated virtual environment detected.

   This script installs LMCache in editable mode and must not write to the
   system Python. Create and activate one first:

       uv venv --python 3.12 && source .venv/bin/activate
       # or: python3 -m venv .venv && source .venv/bin/activate

   Then re-run this script.
MSG
  exit 1
fi

# ------------------------------------------------------------------ #
# Preflight 2: system libraries vLLM dlopens at import time
# ------------------------------------------------------------------ #
# Recent vLLM nightlies import torchcodec unconditionally, which dlopens
# libavutil -- so `vllm serve` aborts with "libavutil.so.NN: cannot open
# shared object file" without FFmpeg, even for a text-only model. Some CPU
# code paths also want libnuma. CI installs these from its workflow; we only
# check and tell you, because an example has no business running sudo on
# your machine.
MISSING_HINT=""
case "$(uname -s)" in
  Linux)
    NEEDED=()
    if ! ldconfig -p 2>/dev/null | grep -q 'libavutil\.so'; then
      NEEDED+=(ffmpeg)
    fi
    if ! ldconfig -p 2>/dev/null | grep -q 'libnuma\.so\.1'; then
      NEEDED+=(libnuma1)
    fi
    if [ "${#NEEDED[@]}" -gt 0 ]; then
      MISSING_HINT="sudo apt-get update && sudo apt-get install -y ${NEEDED[*]}"
    fi
    ;;
  Darwin)
    if ! command -v ffmpeg >/dev/null 2>&1; then
      MISSING_HINT="brew install ffmpeg"
    fi
    ;;
esac
if [ -n "${MISSING_HINT}" ]; then
  cat >&2 <<MSG
!! Missing system libraries that vLLM loads at import time.

   Tier 1 (\`vllm serve\`) will fail without them. Install with:

       ${MISSING_HINT}

   Then re-run this script. Tier 0 does not need vLLM and will work
   regardless.
MSG
  exit 1
fi
echo "==> System libraries present (FFmpeg/libavutil)"

# ------------------------------------------------------------------ #
# 1. vLLM CPU build  2. LMCache CPU build
# ------------------------------------------------------------------ #
echo ""
echo "==> [1/3] Installing vLLM CPU build"
bash "${CI_SCRIPTS}/install_vllm_cpu.sh"

echo ""
echo "==> [2/3] Installing LMCache (NO_GPU_EXT=1)"
bash "${CI_SCRIPTS}/install_lmcache_cpu.sh"

# ------------------------------------------------------------------ #
# 3. Verify
# ------------------------------------------------------------------ #
echo ""
echo "==> [3/3] Verifying the CPU install"
python3 - <<'PY'
import torch
import vllm

import lmcache

print(f"    lmcache: {lmcache.__version__}")
print(f"    vllm:    {vllm.__version__}")
print(f"    torch:   {torch.__version__}")
print(f"    cuda available: {torch.cuda.is_available()}  (expected: False)")
PY

echo ""
echo "==> CPU install complete. Next:"
echo "    examples/cpu_hello_world/tier0_storage_smoke/run.sh   # no model, no GPU"
echo "    examples/cpu_hello_world/tier1_ttft_cpu/run_demo.sh   # tiny model on CPU"
