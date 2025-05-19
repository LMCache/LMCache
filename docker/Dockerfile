# The LMcache Dockerfile is used to build a LMCache image that is integrated
# to run with vLLM OpenAI server.

# Please update any changes made here to:
# docs/source/getting_started/installation.rst

ARG CUDA_VERSION=12.8.1
ARG VLLM_IMAGE_REPO='vllm/vllm-openai'
ARG VLLM_TAG='latest'
ARG UBUNTU_VERSION=22.04

#################### BASE BUILD IMAGE ####################
# Prepare basic build environment

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache software-properties-common git curl sudo \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} \
    && python3 --version && python3 -m pip --version

WORKDIR /workspace

# install runtime dependencies
COPY requirements.txt requirements.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-cuda.txt


# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# Override the arch list for flash-attn to reduce the binary size
ARG vllm_fa_cmake_gpu_arches='80-real;90-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}

#################### BUILD IMAGE ####################
# Build lmcache

FROM base AS build

# install build dependencies
COPY requirements-build.txt requirements-build.txt

# max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}

# number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt

ARG LMCACHE_COMMIT_ID=1

RUN git clone https://github.com/LMCache/LMCache.git

WORKDIR /workspace/LMCache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python3 setup.py bdist_wheel --dist-dir=dist_lmcache

#################### vLLM IMAGE ####################
# Integrate vLLM and LMCache and expose vLLM 
# OpenAI server

FROM ${VLLM_IMAGE_REPO}:${VLLM_TAG} AS vllm-openai

ARG VLLM_IMAGE_REPO
ARG VLLM_TAG

RUN --mount=type=bind,from=build,src=/workspace/LMCache/dist_lmcache,target=/vllm-workspace/dist_lmcache \
--mount=type=cache,target=/root/.cache/pip \
pip install dist_lmcache/*.whl --verbose

ENTRYPOINT ["vllm", "serve"]
