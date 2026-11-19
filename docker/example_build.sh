#!/usr/bin/env bash
# Example script to build the LMCache integrated with vLLM container image

# Update the following variables accordingly
# CUDA_VERSION and NGC_VERSION change together: the base image
# nvcr.io/nvidia/cuda-dl-base:<NGC_VERSION>-cuda<CUDA_VERSION>-devel-ubuntu24.04
# must exist (CUDA 13.2: 26.05; CUDA 13.0: 25.09).
CUDA_VERSION="${CUDA_VERSION:-13.2}"
NGC_VERSION="${NGC_VERSION:-26.05}"
DOCKERFILE_NAME='Dockerfile'
# Set VLLM_VERSION to a vLLM release before running this script. Releases up to
# 0.34.0 are built for CUDA 13.0, e.g.:
#   VLLM_VERSION=0.34.0 CUDA_VERSION=13.0 NGC_VERSION=25.09 ./example_build.sh
# Defaults to "nightly" (vLLM main, CUDA 13.2) if not set.
VLLM_VERSION="${VLLM_VERSION:-nightly}"
DOCKER_BUILD_PATH='../' # This path should point to the LMCache root for access to 'requirements' directory
UBUNTU_VERSION=24.04

# `image-build` target will use the latest LMCache and vLLM code
# Change to 'image-release' target for using release package versions of vLLM and LMCache
BUILD_TARGET=image-build 

IMAGE_TAG='lmcache/vllm-openai:build-latest' # Name of container image to build

docker build \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg NGC_VERSION=$NGC_VERSION \
    --build-arg UBUNTU_VERSION=$UBUNTU_VERSION \
    --build-arg VLLM_VERSION=$VLLM_VERSION \
    --target $BUILD_TARGET --file $DOCKERFILE_NAME \
    --tag $IMAGE_TAG  $DOCKER_BUILD_PATH
