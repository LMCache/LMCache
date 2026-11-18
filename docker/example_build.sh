#!/usr/bin/env bash
# Example script to build the LMCache integrated with vLLM container image

# Update the following variables accordingly
CUDA_VERSION=13.0
DOCKERFILE_NAME='Dockerfile'
# Set VLLM_VERSION to pick vLLM, e.g. VLLM_VERSION=0.34.0 ./example_build.sh.
# "nightly" installs vLLM main, which is published only for CUDA 13.2 (set
# CUDA_VERSION=13.2 and use a driver that supports it).
# If not set, the Dockerfile default (newest vLLM release for CUDA 13.0) is used.
VLLM_VERSION="${VLLM_VERSION:-}"
DOCKER_BUILD_PATH='../' # This path should point to the LMCache root for access to 'requirements' directory
UBUNTU_VERSION=24.04

# `image-build` target will use the latest LMCache and vLLM code
# Change to 'image-release' target for using release package versions of vLLM and LMCache
BUILD_TARGET=image-build 

IMAGE_TAG='lmcache/vllm-openai:build-latest' # Name of container image to build

docker build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg UBUNTU_VERSION=$UBUNTU_VERSION \
    ${VLLM_VERSION:+--build-arg VLLM_VERSION=$VLLM_VERSION} \
    --target $BUILD_TARGET --file $DOCKERFILE_NAME \
    --tag $IMAGE_TAG  $DOCKER_BUILD_PATH
