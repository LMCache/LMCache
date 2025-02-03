# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.4.1
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS base
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.12
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

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements-common.txt requirements-common.txt
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
#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
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
RUN git clone https://github.com/LMCache/torchac_cuda.git


WORKDIR /workspace/LMCache
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python3 setup.py bdist_wheel --dist-dir=dist_lmcache

WORKDIR /workspace/torchac_cuda
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    python3 setup.py bdist_wheel --dist-dir=/workspace/LMCache/dist_lmcache
    

#################### vLLM installation IMAGE ####################
# Install torchac_cuda wheel into the vLLM image
FROM vllm/vllm-openai:v0.7.0 AS vllm-openai
RUN --mount=type=bind,from=build,src=/workspace/LMCache/dist_lmcache,target=/vllm-workspace/dist_lmcache \
--mount=type=cache,target=/root/.cache/pip \
pip install dist_lmcache/*.whl --verbose

# Copy lmc_connector patch into vllm
COPY patch/factory.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/
COPY patch/lmcache_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/

# Use diff if file is too large
COPY patch/parallel_state.patch \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/

RUN patch /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/parallel_state.patch


ENTRYPOINT ["vllm", "serve"]
