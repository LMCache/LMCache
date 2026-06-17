#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Download facebook/opt-125m from the LMCache/opt-125m GitHub release
# and place it into the HuggingFace hub cache structure so vLLM can
# find it via ``facebook/opt-125m`` without any network call to HF.
#
# Environment:
#   OPT125M_GH_RELEASE  release tag (default: v1.0)
#   OPT125M_SNAPSHOT    HF snapshot hash (default: 27dcfa...)

set -euo pipefail

GH_RELEASE="${OPT125M_GH_RELEASE:-v1.0}"
SNAPSHOT="${OPT125M_SNAPSHOT:-27dcfa74d334bc871f3234de431e71c6eeba5dd6}"

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_DIR="${HF_CACHE}/models--facebook--opt-125m"
SNAPSHOT_DIR="${MODEL_DIR}/snapshots/${SNAPSHOT}"

if [ -f "${SNAPSHOT_DIR}/pytorch_model.bin" ]; then
    echo "opt-125m already cached in ${SNAPSHOT_DIR}, skipping download"
    exit 0
fi

echo "Downloading opt-125m from GitHub release ${GH_RELEASE}..."
mkdir -p "${SNAPSHOT_DIR}"

TARBALL="$(mktemp -d)/opt-125m.tar.gz"
DOWNLOAD_URL="https://github.com/LMCache/opt-125m/archive/refs/tags/${GH_RELEASE}.tar.gz"

for i in $(seq 1 5); do
    if curl -fsSL --retry 3 -o "${TARBALL}" "${DOWNLOAD_URL}"; then
        break
    fi
    if [ "${i}" -eq 5 ]; then
        echo "!! Failed to download after 5 attempts: ${DOWNLOAD_URL}"
        exit 1
    fi
    sleep $((10 * i))
done

# Extract to temp dir first, then move files. GitHub archives
# produce a top-level dir like ``opt-125m-1.0``.
EXTRACT_DIR="$(mktemp -d)"
tar -xzf "${TARBALL}" -C "${EXTRACT_DIR}"
# Find the inner directory (only one should exist).
INNER_DIR="$(find "${EXTRACT_DIR}" -mindepth 1 -maxdepth 1 -type d | head -1)"
mv "${INNER_DIR}"/* "${SNAPSHOT_DIR}/"

rm -rf "${TARBALL}" "${EXTRACT_DIR}"

# Create the refs/main pointer so snapshot_download(local_files_only=True)
# can resolve the snapshot hash.
echo "${SNAPSHOT}" > "${MODEL_DIR}/refs/main"

echo "opt-125m cached at ${SNAPSHOT_DIR}"
ls -lah "${SNAPSHOT_DIR}/"