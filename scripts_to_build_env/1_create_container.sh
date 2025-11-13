#!/bin/sh
# Create a container for LMCache.
# Usage:
#   IMAGE=repo/image:tag ./0_create_container.sh
#
# Defaults:
#   IMAGE defaults to lmcache:vllm-openai
IMAGE="${IMAGE:-lmcache:1.1}"

docker run -itd --name lmcache --gpus all --ipc=host \
  --entrypoint bash \
  -v "$HOME/workspace":/root/workspace \
  -v "$HOME/.cache":/root/.cache \
  "$IMAGE"