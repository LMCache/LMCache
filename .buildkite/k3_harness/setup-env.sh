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

# Probe the vLLM CLI import chain. This is the same chain that `vllm serve`
# runs at startup, so anything that fails here would otherwise surface 180s
# later as a silent "vLLM failed to start on port 8000" timeout in the test
# harness. Shared between the pre-install auto-heal loop below and the
# post-install hard probe at the end of this script.
probe_vllm_cli() {
    python -c "from vllm.entrypoints.cli.main import main" 2>&1
}

# vLLM nightlies periodically add eager imports of packages that aren't in
# their declared deps (e.g. `pandas` from vllm/_aiter_ops.py). Auto-install
# any ModuleNotFoundError modules so the job keeps going. Capped to avoid
# infinite loops; every auto-install is logged so the drift is visible in
# the build output. ImportError with a missing top-level name (e.g. a
# transformers/vLLM API break) bails immediately since reinstalling the
# package wouldn't recover.
MAX_AUTO_INSTALL=5
for i in $(seq 1 "$MAX_AUTO_INSTALL"); do
    if err=$(probe_vllm_cli); then
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
# Snapshot env before/after so silent downgrades triggered by LMCache's
# transitive pins (requirements/common.txt caps opentelemetry-*, prometheus,
# etc.) are visible in the build log. Without this, a version-cap-induced
# downgrade can leave /opt/venv in a state that passes the pre-install CLI
# probe but breaks at `vllm serve` time, which is the failure mode that
# motivated this script's post-install hard probe below.
uv pip freeze | sort > /tmp/env-before-lmcache.txt
uv pip install -e . --no-build-isolation
uv pip freeze | sort > /tmp/env-after-lmcache.txt
if ! diff -q /tmp/env-before-lmcache.txt /tmp/env-after-lmcache.txt >/dev/null; then
    echo "--- :warning: Packages changed during LMCache install"
    diff /tmp/env-before-lmcache.txt /tmp/env-after-lmcache.txt || true
fi

echo "--- :mag: Post-install CLI chain probe"
# The LMCache editable install can downgrade transitive deps to honor the
# caps in requirements/common.txt. If that leaves the env in a state where
# the vLLM CLI import chain (vllm.entrypoints.cli.main → vllm.config →
# vllm.transformers_utils.config → `from transformers import ...`) fails,
# the only other signal is a 180s `wait_for_server` timeout inside each
# test harness. Re-probe the full chain here so broken envs fail fast with
# the actual traceback instead of a generic timeout.
if err=$(probe_vllm_cli); then
    echo "vLLM CLI import chain OK post-install."
else
    echo "FATAL: vLLM CLI import chain broken after LMCache install." >&2
    echo "--- Traceback ---" >&2
    echo "$err" >&2
    echo "--- Installed packages ---" >&2
    uv pip freeze >&2 || true
    exit 1
fi

echo "--- :white_check_mark: Environment ready"
python -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation')"
