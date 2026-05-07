#!/usr/bin/env bash
# Per-job environment setup: installs vLLM (nightly cu128 wheels) + LMCache from source.
# Called at the start of every CI job.
set -euo pipefail

# Print the failing command and line number on any error.
trap 'echo "ERROR: setup-blend-env.sh failed at line $LINENO (exit code $?)" >&2' ERR

# ── GPU health pre-check ────────────────────────────────────
# Fail fast if GPUs are occupied by stale host processes.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Bind-mounted repos are often owned by the host user while this script runs as root in Docker;
# git then refuses "dubious ownership" and setuptools_scm fails during editable installs.
if command -v git &>/dev/null; then
    git config --global --add safe.directory "${REPO_ROOT}" 2>/dev/null || true
fi
source "${REPO_ROOT}/.buildkite/k3_tests/common_scripts/helpers.sh"
check_gpu_health 80

echo "--- :python: Installing vLLM (nightly wheels)"


DEFAULT_VENV_BIN="/opt/venv/bin"
UV_BIN="$(command -v uv 2>/dev/null || true)"
UV_BIN="${UV_BIN:-/usr/local/bin/uv}"

if [[ ! -x "${DEFAULT_VENV_BIN}/python" ]]; then
    echo "ERROR: default venv python missing or not executable: ${DEFAULT_VENV_BIN}/python" >&2
    exit 1
fi

vllm_default_out="$("${DEFAULT_VENV_BIN}/python" -c "import vllm; print(vllm.__version__)" 2>&1)" || {
    echo "ERROR: vLLM is not importable in default venv (${DEFAULT_VENV_BIN}). Diagnostics:" >&2
    echo "  python: $("${DEFAULT_VENV_BIN}/python" -c "import sys; print(sys.executable)" 2>&1)" >&2
    echo "  import attempt output:" >&2
    echo "${vllm_default_out}" >&2
    exit 1
}
echo "vLLM in default venv (${DEFAULT_VENV_BIN}): ${vllm_default_out}"

# If uv prompts because /workspace/.venv already exists: use the `--clear` flag or set UV_VENV_CLEAR=1
# to skip the prompt and recreate; this script defaults to --allow-existing (reuse, non-interactive).
UV_VENV_CLEAR="${UV_VENV_CLEAR:-0}"
mkdir -p /workspace
if [[ "${UV_VENV_CLEAR}" == "1" ]]; then
    echo "[HINT] UV_VENV_CLEAR=1: recreating /workspace/.venv with --clear (no prompt)."
    "${UV_BIN}" venv --clear /workspace/.venv --python "${DEFAULT_VENV_BIN}/python3.12" --seed
else
    "${UV_BIN}" venv --allow-existing /workspace/.venv --python "${DEFAULT_VENV_BIN}/python3.12" --seed
fi
TEST_VENV_BIN="/workspace/.venv/bin"

# When flashinfer and flashinfer-cubin resolve to different patch versions, skip strict check.
export FLASHINFER_DISABLE_VERSION_CHECK=1

"${UV_BIN}" pip install -p "${TEST_VENV_BIN}/python" -U vllm --pre \
    --extra-index-url "https://wheels.vllm.ai/nightly/cu129" \
    --extra-index-url "https://download.pytorch.org/whl/cu129" \
    --index-strategy unsafe-best-match


"${DEFAULT_VENV_BIN}/python" -c 'import vllm; print(f"default venv vllm={vllm.__version__}")'
"${TEST_VENV_BIN}/python" -c 'import vllm; print(f"test venv vllm={vllm.__version__}")'
"${DEFAULT_VENV_BIN}/python" -c 'import torch; print(f"default venv torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")'
"${TEST_VENV_BIN}/python" -c 'import torch; print(f"test venv torch={torch.__version__}, torch.version.cuda={torch.version.cuda}")'

echo "--- :wrench: Install sitecustomize.py in both venvs (transformers pre-import)"
# The blend test runs prefiller in /opt/venv and decoder in
# /workspace/.venv. Both launch vllm, which spawns a BG thread that
# calls `import transformers` concurrently with the main thread's
# later `from transformers import GenerationConfig, ...`. Pre-import
# on the main thread via sitecustomize.py so the race is closed in
# both venvs (see setup-env.sh for the full write-up).
for site in \
    /opt/venv/lib/python3.12/site-packages \
    /workspace/.venv/lib/python3.12/site-packages; do
    cat > "${site}/sitecustomize.py" <<'PY'
try:
    import transformers  # noqa: F401
except Exception:
    pass
PY
done

echo "--- :python: Installing LMCache from source"
# Skip setuptools_scm git describe; the repo carries non-PEP-440 tags
# (nightly, nightly-cu13) that crash the newer vcs_versioning backend.
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-0.0.0+ci}"
"${UV_BIN}" pip install -p "${DEFAULT_VENV_BIN}/python" -e . --no-build-isolation
"${UV_BIN}" pip install -p "${TEST_VENV_BIN}/python" -e . --no-build-isolation
# Work around openai_harmony vocab download/load issues for GPT-OSS (vLLM recipes troubleshooting).
# related github issue: https://github.com/openai/harmony/pull/41
TIKTOKEN_ENCODINGS_DIR="${REPO_ROOT}/tiktoken_encodings"
mkdir -p "${TIKTOKEN_ENCODINGS_DIR}"
if ! command -v curl &>/dev/null; then
    echo "ERROR: curl is required for downloading tiktoken encodings" >&2
    exit 1
fi
if [[ ! -s "${TIKTOKEN_ENCODINGS_DIR}/o200k_base.tiktoken" ]]; then
  curl -fsSL "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" -o "${TIKTOKEN_ENCODINGS_DIR}/o200k_base.tiktoken"
fi
if [[ ! -s "${TIKTOKEN_ENCODINGS_DIR}/cl100k_base.tiktoken" ]]; then
  curl -fsSL "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" -o "${TIKTOKEN_ENCODINGS_DIR}/cl100k_base.tiktoken"
fi


export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_DIR}"
echo "Using TIKTOKEN_ENCODINGS_BASE=${TIKTOKEN_ENCODINGS_BASE}"


echo "--- :white_check_mark: Environment ready"
"${DEFAULT_VENV_BIN}/python" -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation in default venv')"
"${TEST_VENV_BIN}/python" -c "import vllm; import lmcache; print(f'vLLM={vllm.__version__}, LMCache installed from source with no build isolation in test venv')"
