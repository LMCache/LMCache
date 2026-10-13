#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <image>:<tag> [<image>:<tag> ...]" >&2
    exit 2
fi

volume=""
cleanup() {
    if [ -n "$volume" ]; then
        docker volume rm "$volume" >/dev/null
    fi
}
trap cleanup EXIT

for image in "$@"; do
    docker run --rm --entrypoint /bin/bash "$image" -ec '
        test "$(id -u)" = 1000
        test "$(id -un)" = lmcache
        test "$HOME" = /home/lmcache
        test -x /opt/venv/bin/python3
        test ! -w /opt/venv
        touch /workspace/nonroot-probe
        mkdir -p "$HOME/.cache/vllm"
        touch "$HOME/.cache/vllm/nonroot-probe"
        touch /dev/shm/nonroot-probe
    '

    volume=$(docker volume create)
    docker run --rm --entrypoint /bin/bash \
        -v "$volume":/home/lmcache/.cache/huggingface "$image" -ec '
        test "$(id -u)" = 1000
        mkdir -p "$HOME/.cache/huggingface/hub"
        touch "$HOME/.cache/huggingface/hub/nonroot-probe"
        mkdir -p "$HOME/.cache/vllm"
        touch "$HOME/.cache/vllm/nonroot-probe"
    '
    cleanup
    volume=""
    echo "Non-root smoke test passed: $image"
done
