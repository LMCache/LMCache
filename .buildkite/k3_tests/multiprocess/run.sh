#!/usr/bin/env bash
# Multiprocessing test for K8s pods.
#
# This test builds Docker images and launches containers (LMCache MP server,
# vLLM with LMCache, vLLM baseline). It requires Docker-in-Docker: the pod
# mounts the host's Docker socket.
#
# The old script used pick-free-gpu.sh for GPU selection. In K8s, the
# device plugin assigns GPUs, and CUDA_VISIBLE_DEVICES is already set.
# We pre-set the GPU variables so the launch script skips pick-free-gpu.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# ── Environment setup ────────────────────────────────────────
source .buildkite/k3_harness/setup-env.sh

# ── Verify Docker access ────────────────────────────────────
if ! docker info &>/dev/null; then
    echo "Docker not available in pod. Ensure docker.sock is mounted."
    exit 1
fi

# ── Run the existing MP test orchestrator ────────────────────
# The existing scripts handle build, launch, wait, test, cleanup.
export BUILD_ID="${BUILDKITE_BUILD_ID:-local_$$}"

chmod +x .buildkite/scripts/multiprocessing-test/*.sh
BUILD_ID="${BUILD_ID}" .buildkite/scripts/multiprocessing-test/run-mp-test.sh
