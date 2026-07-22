#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Download a HuggingFace-shaped model snapshot from a GitHub Release
# tarball and populate the local HF hub cache so downstream tooling
# (vLLM, transformers, ...) resolves the model via its normal HF repo
# id without any network call to HuggingFace.
#
# This is the CPU CI's way of bypassing HF rate limits / outages for
# small helper models. The tarball is expected to already have the
# HF snapshot layout at its top-level directory.
#
# Required env vars:
#   MODEL_ID          HF repo id, e.g. facebook/opt-125m
#                     Used to derive the HF cache directory name
#                     (org/name -> models--org--name).
#   TARBALL_URL       Full URL to a tar.gz whose top-level directory is
#                     ${TARBALL_TOP_DIR}. Works with any host (GitHub
#                     release asset, GitHub branch/tag archive, S3, ...).
#                     Examples:
#                     https://github.com/OWNER/REPO/releases/download/TAG/FILE.tar.gz
#                     https://github.com/OWNER/REPO/archive/refs/heads/BRANCH.tar.gz
#                     https://github.com/OWNER/REPO/archive/refs/tags/TAG.tar.gz
#   SNAPSHOT          HF snapshot hash (populates snapshots/<hash>/
#                     and refs/main)
#   TARBALL_TOP_DIR   Directory name inside the tarball whose contents
#                     should be moved into snapshots/<hash>/
#                     (e.g. opt-125m, deepseek_v2_lite-with_out_weight)
#
# Optional env vars:
#   CACHE_MARKER      File under snapshots/<hash>/ that, if present,
#                     signals "already cached, skip download". Default:
#                     config.json (works for weight-less mirrors too).

set -euo pipefail

: "${MODEL_ID:?MODEL_ID is required (e.g. facebook/opt-125m)}"
: "${TARBALL_URL:?TARBALL_URL is required (full https URL to the tar.gz)}"
: "${SNAPSHOT:?SNAPSHOT is required (HF snapshot hash)}"
: "${TARBALL_TOP_DIR:?TARBALL_TOP_DIR is required}"
CACHE_MARKER="${CACHE_MARKER:-config.json}"

# HF cache dir name: HF's rule is "models--<org>--<name>", i.e. the
# repo id's "/" separator becomes "--" and every other character is
# kept verbatim. Do NOT touch pre-existing dashes in org/name (e.g.
# "deepseek-ai" must stay as-is, not become "deepseek--ai").
HF_DIR_NAME="models--${MODEL_ID/\//--}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_DIR="${HF_CACHE}/${HF_DIR_NAME}"
SNAPSHOT_DIR="${MODEL_DIR}/snapshots/${SNAPSHOT}"

if [ -f "${SNAPSHOT_DIR}/${CACHE_MARKER}" ]; then
    echo "${MODEL_ID} already cached in ${SNAPSHOT_DIR}, skipping download"
    exit 0
fi

echo "Downloading ${MODEL_ID} from ${TARBALL_URL}..."
mkdir -p "${SNAPSHOT_DIR}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

TARBALL="${TMP_DIR}/model.tar.gz"

for i in $(seq 1 5); do
    if curl -fsSL --retry 3 -o "${TARBALL}" "${TARBALL_URL}"; then
        break
    fi
    if [ "${i}" -eq 5 ]; then
        echo "!! Failed to download after 5 attempts: ${TARBALL_URL}"
        exit 1
    fi
    sleep $((10 * i))
done

# Tarball layout: top-level directory named ${TARBALL_TOP_DIR}
# whose contents are the HF snapshot files.
EXTRACT_DIR="${TMP_DIR}/extract"
mkdir -p "${EXTRACT_DIR}"
tar -xzf "${TARBALL}" -C "${EXTRACT_DIR}"
if [ ! -d "${EXTRACT_DIR}/${TARBALL_TOP_DIR}" ]; then
    echo "!! Tarball from ${TARBALL_URL} does not contain top-level dir '${TARBALL_TOP_DIR}'"
    echo "   Actual contents:"
    ls -la "${EXTRACT_DIR}"
    exit 1
fi
mv "${EXTRACT_DIR}/${TARBALL_TOP_DIR}"/* "${SNAPSHOT_DIR}/"

# Create the refs/main pointer so snapshot_download(local_files_only=True)
# can resolve the snapshot hash.
mkdir -p "${MODEL_DIR}/refs"
echo "${SNAPSHOT}" > "${MODEL_DIR}/refs/main"

echo "${MODEL_ID} cached at ${SNAPSHOT_DIR}"
ls -lah "${SNAPSHOT_DIR}/"
