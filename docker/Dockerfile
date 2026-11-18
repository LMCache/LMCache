# The LMCache Dockerfile is used to build a LMCache image that is integrated
# to run with vLLM OpenAI server.

# Please update any changes made here to
# docs/source/developer_guide/docker_file.rst
# docs/source/getting_started/installation.rst
# docs/source/production/docker_deployment.rst

ARG CUDA_VERSION=13.0
ARG UBUNTU_VERSION=24.04
ARG NGC_VERSION=25.09
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:${NGC_VERSION}-cuda${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

#################### BASE BUILD IMAGE ####################
# Prepare basic build environment

FROM ${BASE_IMAGE} AS base

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.12
ARG UBUNTU_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python and other dependencies
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends \
        ccache software-properties-common git curl sudo \
        python3 python3-dev python3-venv python3-pip tzdata libxcb1-dev \
    && ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/ \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv /usr/local/bin/ \
    && mv ~/.local/bin/uvx /usr/local/bin/ \
    && uv venv /opt/venv \
    && . /opt/venv/bin/activate \
    && python3 --version

WORKDIR /workspace

# The NGC Ubuntu base reserves uid 1000 for its default ubuntu account.
# Replace that account with a dedicated user to reduce container privileges.
RUN if [ "$(getent passwd 1000 | cut -d: -f1)" = "ubuntu" ]; then userdel ubuntu; fi \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin lmcache \
    && mkdir -p /home/lmcache/.cache/huggingface \
    && chown -R lmcache:lmcache /workspace /home/lmcache/.cache

# CUDA arch list used by torch
ARG torch_cuda_arch_list='7.5 8.0 8.6 8.9 9.0 10.0 12.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

#################### vLLM IMAGE & LMCache (Build) ##########################
# Integrate vLLM and an LMCache build from source, and expose vLLM OpenAI API

FROM base AS image-build

# install build dependencies
COPY ./requirements/build.txt build.txt

# Max jobs used by Ninja to build extensions
ARG max_jobs=2
ENV MAX_JOBS=${max_jobs}

# Number of threads used by nvcc
ARG nvcc_threads=auto
ENV NVCC_THREADS=$nvcc_threads

ARG CUDA_VERSION
# vLLM to pair with LMCache. The default is the newest vLLM release with a
# CUDA 13.0 (cu130) build, so the image runs on R580 drivers. vLLM main
# publishes nightly wheels only for cu132 since 2026-11-10; those need a
# CUDA 13.2 driver. Bump this when a newer release ships a cu130 build
# (the nightly workflow warns when it does).
# VLLM_VERSION=nightly installs vLLM main from wheels.vllm.ai/nightly/<cuda>.
ARG VLLM_VERSION=0.34.0
# With VLLM_VERSION=nightly, fail when the nightly index for CUDA_VERSION is
# older than this many days, i.e. vLLM stopped building it (0 disables).
ARG VLLM_NIGHTLY_MAX_AGE_DAYS=7

RUN --mount=type=cache,target=/root/.cache/uv,id=uv-cache,sharing=locked \
    . /opt/venv/bin/activate && \
    uv pip install -r build.txt

ARG LMCACHE_COMMIT_ID=1

COPY . /workspace/LMCache
WORKDIR /workspace/LMCache

 
# vLLM comes from its per-CUDA index (wheels.vllm.ai/<version or nightly>/
# <cuda_tag>) and torch from the matching PyTorch index, so both are built for
# CUDA_VERSION even when PyPI's default vLLM wheel targets another CUDA.
RUN --mount=type=cache,target=/root/.cache/ccache,id=ccache \
    --mount=type=cache,target=/root/.cache/uv,id=uv-cache,sharing=locked \
    . /opt/venv/bin/activate && \
    CUDA_TAG=cu$(echo ${CUDA_VERSION} | tr -d '.') && \
    export LMCACHE_CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) && \
    if [ "$VLLM_VERSION" = "nightly" ]; then \
        VLLM_INDEX=https://wheels.vllm.ai/nightly/${CUDA_TAG} && \
        VLLM_SPEC='vllm[runai,tensorizer,flashinfer]' && \
        python3 .github/scripts/check_vllm_nightly_age.py \
            "${VLLM_INDEX}" "${VLLM_NIGHTLY_MAX_AGE_DAYS}" ; \
    else \
        VLLM_INDEX=https://wheels.vllm.ai/${VLLM_VERSION}/${CUDA_TAG} && \
        VLLM_SPEC="vllm[runai,tensorizer,flashinfer]==${VLLM_VERSION}" ; \
    fi && \
    VLLM_PRECOMPILED_WHEEL_VARIANT=${CUDA_TAG} uv pip install --prerelease=allow \
        "${VLLM_SPEC}" \
        --extra-index-url "${VLLM_INDEX}" \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        --index-strategy unsafe-first-match && \
    uv pip install "nixl<2" && \
    python3 -c 'import importlib.metadata as m; print("VLLM=", m.version("vllm"))' && \
    python3 -c 'import torch; print("TORCH=", torch.__version__)' && \
    python3 -c 'import importlib.metadata as m; print("NIXL=", m.version("nixl"))' && \
    python3 setup.py bdist_wheel --dist-dir=dist_lmcache && \
    bash docker/build_raw_block_wheel.sh dist_lmcache && \
    bash docker/build_nixl_wheel.sh dist_lmcache && \
    uv pip install ./dist_lmcache/*.whl --verbose && \
    python3 .github/scripts/assert_native_ops.py && \
    python3 -c 'import lmcache_rust_raw_block_io' && \
    uv pip install -r requirements/grpc_native.txt && \
    python3 -c 'import pygrpc_native' && \
    uv pip install -r requirements/kafka.txt && \
    python3 -c 'import confluent_kafka' && \
    python3 -c 'import nixl, lmcache_nixl'

WORKDIR /workspace
USER lmcache
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]

#################### vLLM IMAGE & LMCache (Release, cu13) #######################
# Integrate vLLM and LMCache stable releases, and expose vLLM OpenAI API.
# The default lmcache wheel on PyPI is built against cu13; the cu13 torch
# index is hinted explicitly so vLLM also resolves to its cu13 build.

FROM base AS image-release

ARG CUDA_VERSION

ARG LMCACHE_VERSION

RUN --mount=type=bind,source=requirements/grpc_native.txt,target=/tmp/grpc_native.txt \
    --mount=type=bind,source=requirements/kafka.txt,target=/tmp/kafka.txt \
    --mount=type=bind,source=.github/scripts/assert_native_ops.py,target=/tmp/assert_native_ops.py \
    . /opt/venv/bin/activate && \
    CUDA_TAG=cu$(echo ${CUDA_VERSION} | tr -d '.') && \
    CUDA_MAJOR=$(echo ${CUDA_VERSION} | cut -d. -f1) && \
    VER=$(echo "${LMCACHE_VERSION}" | sed 's/^v//') && \
    VLLM_PRECOMPILED_WHEEL_VARIANT=${CUDA_TAG} uv pip install --prerelease=allow \
        vllm[runai,tensorizer,flashinfer] \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        --index-strategy unsafe-best-match && \
    # The nixl extra adds the NIXL connector (lmcache-nixl) at lmcache's version.
    uv pip install "lmcache[nixl]${VER:+==${VER}}" \
        --extra-index-url https://download.pytorch.org/whl/${CUDA_TAG} \
        --index-strategy unsafe-best-match --verbose && \
    # The PyPI wheel is compiled against pyproject's torch pin, not the torch
    # vLLM just installed; fail here if they drifted (cuda_ops ABI mismatch).
    python3 /tmp/assert_native_ops.py && \
    # The Rust module is published at lmcache's version; pin it to the lmcache
    # actually installed, so the two match even when LMCACHE_VERSION is unset.
    LMCACHE_INSTALLED=$(python3 -c 'import importlib.metadata as m; print(m.version("lmcache"))') && \
    uv pip install "lmcache-rust-raw-block-io==${LMCACHE_INSTALLED}" && \
    python3 -c 'import lmcache_rust_raw_block_io' && \
    uv pip install -r /tmp/grpc_native.txt && \
    python3 -c 'import pygrpc_native' && \
    uv pip install -r /tmp/kafka.txt && \
    python3 -c 'import confluent_kafka' && \
    python3 -c 'import nixl, lmcache_nixl'

WORKDIR /workspace
USER lmcache
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]
