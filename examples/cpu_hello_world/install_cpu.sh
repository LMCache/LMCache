#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install LMCache + vLLM in CPU-only mode for the cpu_hello_world demo.
#
# This mirrors the CPU install that LMCache's own CI exercises on GPU-less
# Ubuntu and macOS runners (.github/workflows/cpu_device.yml), packaged as a
# single user-facing script. It does NOT require a GPU, CUDA, or nvcc.
#
# What it installs:
#   1. vLLM CPU build (vllm-cpu-nightly) + its CPU torch wheel, plus the
#      load-bearing "+cpu" dist-info alias (see the note below) so that
#      `vllm serve` activates its CPU platform.
#   2. LMCache built with NO_GPU_EXT=1 (keeps the common C++ extensions
#      -- storage manager / redis / fs -- and drops only the CUDA kernels;
#      the pure-Python fallback in lmcache/python_ops_fallback.py handles
#      host-memory allocation and copies).
#
# Usage (run from anywhere inside the LMCache repo):
#   examples/cpu_hello_world/install_cpu.sh
#   PIP_BIN="uv pip" examples/cpu_hello_world/install_cpu.sh   # use uv
#
# Environment (all optional):
#   PIP_BIN            pip front-end (default: "pip"; e.g. "uv pip")
#   VLLM_CPU_SPEC      vLLM CPU package spec (default: "vllm-cpu-nightly").
#                      Pin to a known-good build for reproducibility, e.g.
#                      VLLM_CPU_SPEC="vllm-cpu-nightly==<version>".

set -euo pipefail

PIP_BIN="${PIP_BIN:-pip}"
VLLM_CPU_SPEC="${VLLM_CPU_SPEC:-vllm-cpu-nightly}"

# Resolve and cd to the repo root so `pip install -e .` sees pyproject.toml
# and requirements/.
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
cd "${REPO_ROOT}"
echo "==> Repo root: ${REPO_ROOT}"
echo "==> Python:    $(python3 --version 2>&1 || true)"
echo "==> pip front-end: ${PIP_BIN}"

# ------------------------------------------------------------------ #
# 1. vLLM CPU build (+ CPU torch)
# ------------------------------------------------------------------ #
# `--extra-index-url .../whl/cpu` is required because the wheel pins a torch
# version that only lives on the PyTorch CPU index. `numpy<2` keeps
# scipy/vLLM happy.
echo ""
echo "==> [1/3] Installing vLLM CPU build (${VLLM_CPU_SPEC})"
${PIP_BIN} install --upgrade pip setuptools wheel
${PIP_BIN} install "numpy<2"
${PIP_BIN} install "${VLLM_CPU_SPEC}" \
  --extra-index-url https://download.pytorch.org/whl/cpu

# The wheel installs the `vllm/` package but registers its dist metadata
# under `vllm-cpu-nightly`. Two things break without an alias:
#   * `importlib.metadata.version("vllm")` -> PackageNotFoundError, so the
#     `vllm serve` CLI won't start.
#   * vLLM's `cpu_platform_plugin()` greps the dist metadata for the
#     substring "cpu" to activate the CPU platform.
# So we copy the dist-info to a `vllm-<ver>+cpu.dist-info` alias and rewrite
# Name/Version. This is idempotent.
echo "==> Aliasing vllm-cpu-nightly dist-info -> vllm (+cpu)"
python - <<'PY'
import importlib.metadata as md
import pathlib
import shutil

dist = md.distribution("vllm-cpu-nightly")
ver = dist.version
fake_ver = f"{ver}+cpu"
site_root = pathlib.Path(dist.locate_file(""))
info_name = next(
    p.parts[0]
    for p in (dist.files or [])
    if p.parts and p.parts[0].endswith(".dist-info")
)
src = site_root / info_name
dst = src.with_name(f"vllm-{fake_ver}.dist-info")
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
meta = dst / "METADATA"
txt = meta.read_text()
txt = txt.replace("Name: vllm-cpu-nightly", "Name: vllm", 1)
txt = txt.replace(f"Version: {ver}", f"Version: {fake_ver}", 1)
meta.write_text(txt)
print(f"    aliased {src.name} -> {dst.name}")
PY

# ------------------------------------------------------------------ #
# 2. LMCache (CPU-only build)
# ------------------------------------------------------------------ #
echo ""
echo "==> [2/3] Installing LMCache (NO_GPU_EXT=1, --no-deps to keep CPU torch)"
export NO_GPU_EXT=1
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.0.0.dev0}"
${PIP_BIN} install -r requirements/build.txt
${PIP_BIN} install -r requirements/common.txt
${PIP_BIN} install -r requirements/cli.txt
${PIP_BIN} install -e . --no-deps --no-build-isolation

# ------------------------------------------------------------------ #
# 3. Verify
# ------------------------------------------------------------------ #
echo ""
echo "==> [3/3] Verifying the CPU install"
python - <<'PY'
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
