#!/usr/bin/env bash
# Correctness test for K8s pods.
#
# The old script assumed a persistent venv at $HOME/correctness/.venv and
# a pre-downloaded ShareGPT dataset. Here the pod gets a fresh environment
# from setup-env.sh, and the dataset is mounted from the host volume.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh
uv pip install aiohttp tqdm pandas huggingface_hub

# ── Check prerequisites ──────────────────────────────────────
SHAREGPT_PATH="/root/correctness/.ShareGPT_V3_unfiltered_cleaned_split.json"
if [[ ! -f "$SHAREGPT_PATH" ]]; then
    echo "[INFO] ShareGPT dataset not found, downloading..."
    mkdir -p /root/correctness
    wget -q \
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" \
        -O "$SHAREGPT_PATH"
fi

# ── Run the existing correctness script ──────────────────────
# The script expects BUILD_ID and uses pick-free-gpu.sh.
# In K8s, CUDA_VISIBLE_DEVICES is set by the device plugin,
# so we stub out pick-free-gpu.sh.
export BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"

# Override pick-free-gpu.sh — K8s already assigned our GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p .buildkite/scripts
cat > .buildkite/scripts/pick-free-gpu-k8s-stub.sh <<'STUB'
#!/usr/bin/env bash
# Stub: K8s device plugin already assigned GPUs via CUDA_VISIBLE_DEVICES.
echo "✓ Using K8s-assigned GPU(s): CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
STUB
chmod +x .buildkite/scripts/pick-free-gpu-k8s-stub.sh

# Patch the correctness script's source of pick-free-gpu.sh
# by pre-setting the variables it would set.
# Then run the script directly.
export HOME="/root"
ln -sf "$SHAREGPT_PATH" "$HOME/correctness/.ShareGPT_V3_unfiltered_cleaned_split.json" 2>/dev/null || true

# Create the venv the script expects at $HOME/correctness/.venv
# (the script sources it, so we symlink to the pod's venv)
mkdir -p "$HOME/correctness"
ln -sf /opt/venv "$HOME/correctness/.venv"

bash .buildkite/scripts/vllm-correctness.sh
