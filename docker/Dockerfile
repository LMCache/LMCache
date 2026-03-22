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
ENV PATH="/opt/venv/bin:${PATH}"
ENV LMCACHE_FILTER_TORCH_CUDA_ARCH_LIST=/usr/local/bin/lmcache-filter-torch-cuda-arch-list

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

RUN cat <<'EOF' >/usr/local/bin/lmcache-filter-torch-cuda-arch-list
#!/bin/sh
set -eu

printf '%s\n' "${1:-}" | awk '
BEGIN { first = 1 }
{
    for (i = 1; i <= NF; ++i) {
        if ($i != "7.0" && $i != "7.0+PTX") {
            if (!first) {
                printf " "
            }
            printf "%s", $i
            first = 0
        }
    }
}
END { print "" }'
EOF

RUN chmod +x "${LMCACHE_FILTER_TORCH_CUDA_ARCH_LIST}"

WORKDIR /workspace

# Install runtime dependencies
COPY ./requirements/common.txt common.txt
COPY ./requirements/cuda.txt cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    CUDA_MAJOR="${CUDA_VERSION%%.*}" && \
    . /opt/venv/bin/activate && \
    uv pip install -r cuda.txt && \
    if [ "$CUDA_MAJOR" -ge 13 ]; then \
        uv pip uninstall cupy-cuda12x nixl nixl-cu12 || true && \
        uv pip install cupy-cuda13x nixl-cu13; \
    fi

# CUDA arch list used by torch
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0 10.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}

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

ARG VLLM_VERSION=nightly

RUN --mount=type=cache,target=/root/.cache/pip \
    . /opt/venv/bin/activate && \
    uv pip install -r build.txt

ARG LMCACHE_COMMIT_ID=1

COPY . /workspace/LMCache
WORKDIR /workspace/LMCache

# Build LMCache.
# As we are installing vLLM before LMCache, we will implicitly use
# the same torch version as the vLLM build when running the LMCache build.
# This means that LMCache is in sync with the vLLM torch version.
RUN --mount=type=cache,target=/root/.cache/ccache,id=ccache \
    --mount=type=cache,target=/root/.cache/uv,id=uv-cache,sharing=locked \
    CUDA_MAJOR="${CUDA_VERSION%%.*}" && \
    . /opt/venv/bin/activate && \
    if [ "$VLLM_VERSION" = "nightly" ]; then \
        uv pip install --prerelease=allow \
            'vllm[runai,tensorizer,flashinfer]' \
            'triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels' \
            --extra-index-url https://wheels.vllm.ai/nightly \
            --index-strategy unsafe-best-match ; \
    else \
        uv pip install --prerelease=allow \
            "vllm[runai,tensorizer,flashinfer]==${VLLM_VERSION}" \
            'triton-kernels @ git+https://github.com/triton-lang/triton.git@v3.5.0#subdirectory=python/triton_kernels' ; \
    fi && \
    python3 -c 'import torch; print("TORCH=", torch.__version__)' && \
    LM_TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" && \
    if [ "$CUDA_MAJOR" -ge 13 ]; then \
        LM_TORCH_CUDA_ARCH_LIST="$(${LMCACHE_FILTER_TORCH_CUDA_ARCH_LIST} "${TORCH_CUDA_ARCH_LIST}")"; \
    fi && \
    TORCH_CUDA_ARCH_LIST="${LM_TORCH_CUDA_ARCH_LIST}" \
        python3 setup.py bdist_wheel --dist-dir=dist_lmcache && \
    uv pip install ./dist_lmcache/*.whl --verbose --no-deps

WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]

#################### vLLM IMAGE & LMCache (Release) #######################
# Integrate vLLM and LMCache stable releases, and expose vLLM OpenAI API

FROM base AS image-release

ARG CUDA_VERSION

# Copy LMCache sources so CUDA 13 builds can be compiled when needed.
COPY ./requirements/build.txt /workspace/build.txt
COPY . /workspace/LMCache
WORKDIR /workspace/LMCache

RUN --mount=type=cache,target=/root/.cache/uv \
    . /opt/venv/bin/activate && \
    uv pip install -r /workspace/build.txt

# Install vLLM stable release and LMCache.
# CUDA 12 uses the published LMCache wheel.
# CUDA 13 builds LMCache from source against the image-local CUDA/Torch stack.
RUN . /opt/venv/bin/activate && \
    CUDA_MAJOR="${CUDA_VERSION%%.*}" && \
    CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') && \
    BUILD_PKGS="libcusparse-dev-${CUDA_VERSION_DASH} libcublas-dev-${CUDA_VERSION_DASH} libcusolver-dev-${CUDA_VERSION_DASH}" && \
    uv pip install --prerelease=allow vllm[runai,tensorizer,flashinfer] && \
    if [ "$CUDA_MAJOR" -ge 13 ]; then \
        apt-get update -y && \
        apt-get install -y --no-install-recommends ${BUILD_PKGS} && \
        LM_TORCH_CUDA_ARCH_LIST="$(${LMCACHE_FILTER_TORCH_CUDA_ARCH_LIST} "${TORCH_CUDA_ARCH_LIST}")" && \
        TORCH_CUDA_ARCH_LIST="${LM_TORCH_CUDA_ARCH_LIST}" \
            uv pip install . --verbose --no-build-isolation --no-deps && \
        apt-get purge -y ${BUILD_PKGS} && \
        rm -rf /var/lib/apt/lists/*; \
    else \
        uv pip install lmcache --verbose; \
    fi && \
    rm -rf /workspace/LMCache /workspace/build.txt

WORKDIR /workspace
ENTRYPOINT ["/opt/venv/bin/vllm", "serve"]
