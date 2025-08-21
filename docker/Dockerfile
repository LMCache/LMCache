# The LMCache Dockerfile is used to build a LMCache image that is integrated
# to run with vLLM OpenAI server.

# Please update any changes made here to
# docs/source/developer_guide/docker_file.rst
# docs/source/getting_started/installation.rst
# docs/production/docker_deployment.rst

ARG CUDA_VERSION=12.8
ARG UBUNTU_VERSION=24.04

#################### BASE BUILD IMAGE ####################
# Prepare basic build environment

FROM nvcr.io/nvidia/cuda-dl-base:25.03-cuda${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ccache software-properties-common git curl sudo \
        python3 python3-dev python3-venv python3-pip tzdata \
    && ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/ \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv /usr/local/bin/ \
    && mv ~/.local/bin/uvx /usr/local/bin/ \
    && uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && python3 --version

WORKDIR /workspace

# Install and setup nixl
RUN apt-get update -y && \
    apt-get -y install \
    ninja-build \
    pybind11-dev \
    python${PYTHON_VERSION}-dev \
    cmake
RUN export LD_LIBRARY_PATH=/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
RUN export NIXL_PLUGIN_DIR=/usr/local/nixl/lib/x86_64-linux-gnu/plugins
RUN cd /workspace
RUN git clone https://github.com/ai-dynamo/nixl && \
    cd nixl && \
    git checkout b1c22edd8fe10e2e5221c107ee51200fce6f09a8
RUN cd /workspace/nixl
RUN source /opt/venv/bin/activate
RUN . /opt/venv/bin/activate && \
    uv pip install meson
RUN cd /workspace/nixl && \
    . /opt/venv/bin/activate && \
    rm -rf build && \
    mkdir build && \
    uv run meson setup build/ --prefix=/usr/local/nixl && \
    cd build && \
    ninja && \
    ninja install
RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu" > /etc/ld.so.conf.d/nixl.conf
RUN echo "/usr/local/nixl/lib/x86_64-linux-gnu/plugins" >> /etc/ld.so.conf.d/nixl.conf
RUN ldconfig
RUN cd /workspace/nixl/ && \
    . /opt/venv/bin/activate && \
    uv build --wheel --out-dir /tmp/dist && \
    uv pip install /tmp/dist/nixl-0.3.0-cp312-cp312-linux_x86_64.whl

# Install runtime dependencies
COPY ./requirements/common.txt common.txt
COPY ./requirements/cuda.txt cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/venv/bin/activate && \
    uv pip install -r cuda.txt

# CUDA arch list used by torch
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0 10.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

# Override the arch list for flash-attn to reduce the binary size
ARG vllm_fa_cmake_gpu_arches='80-real;90-real;100-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}

#################### vLLM IMAGE & LMCache (Build) ##########################
# Integrate vLLM nightly build and LMCache build, and expose vLLM OpenAI API

FROM base AS image-build

# install build dependencies
COPY ./requirements/build.txt build.txt

# Max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}

# Number of threads used by nvcc
ARG nvcc_threads=8
ENV NVCC_THREADS=$nvcc_threads

RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/venv/bin/activate && \
    uv pip install -r build.txt

ARG LMCACHE_COMMIT_ID=1

COPY . /workspace/LMCache
WORKDIR /workspace/LMCache

# Build LMCache.
# As we are installing vLLM before LMCache, we will implicitly use
# the same torch version as the vLLM nightly build when running the LMCache build.
# This means that LMCache is in sync with the vLLM torch version.
# Stable vLLM release will be used if nightly build is not available.
RUN --mount=type=cache,target=/root/.cache/ccache,id=ccache \
    --mount=type=cache,target=/root/.cache/uv,id=uv-cache,sharing=locked \
    . /opt/venv/bin/activate && \
    uv pip install --prerelease=allow 'vllm[runai,tensorizer]' \
        --extra-index-url https://wheels.vllm.ai/nightly \
        --index-strategy unsafe-best-match && \
    python3 -c 'import torch; print("TORCH=", torch.__version__)' && \
    python3 setup.py bdist_wheel --dist-dir=dist_lmcache && \
    uv pip install ./dist_lmcache/*.whl --verbose

WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]

#################### vLLM IMAGE & LMCache (Release) #######################
# Integrate vLLM and LMCache stable releases, and expose vLLM OpenAI API

FROM base AS image-release

# Install LMCache and vLLM stable releases.
# It is imperative that LMCache uses the same torch version as the 
# vLLM stable release.
RUN . /opt/venv/bin/activate && \
    uv pip install --prerelease=allow vllm[runai,tensorizer] && \
    uv pip install lmcache --verbose

WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]
