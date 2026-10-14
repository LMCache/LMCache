# PRT images. Build with build.sh.
#
#   serving  stock vLLM nightly wheel of VLLM_COMMIT, no source patches; adds
#            only startup packaging. Shared by every deployment.
#   lmcache  serving + the LMCache under test (`lmcache_src` context); runs
#            both `vllm serve` and `lmcache server`.
#   mooncake serving + the Mooncake nightly wheel; also runs mooncake_master
#            and mooncake_client.
#   aiperf   the load generator.

ARG CUDA_VERSION=13.0
ARG UBUNTU_VERSION=24.04
ARG NGC_VERSION=25.09
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:${NGC_VERSION}-cuda${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Base of the lmcache and mooncake stages; build.sh passes the pinned image.
ARG SERVING_IMAGE=serving

#################### serving ####################

FROM ${BASE_IMAGE} AS serving

ARG CUDA_VERSION
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        git curl patch python3 python3-dev python3-venv libxcb1-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig /usr/local/cuda-${CUDA_VERSION}/compat/ \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv ~/.local/bin/uvx /usr/local/bin/ \
    && uv venv --python 3.12 /opt/venv

# Full 40-char sha; the per-commit index keeps the wheel after nightly moves on.
# LMCache needs protobuf<7.
ARG VLLM_COMMIT
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv,sharing=locked \
    uv pip install --python /opt/venv/bin/python --prerelease=allow \
        'vllm[flashinfer]' grpcio grpcio-tools 'protobuf<7' \
        --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT}/cu130 \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
    && mkdir -p /opt/prt \
    && uv pip freeze --python /opt/venv/bin/python > /opt/prt/serving-freeze.txt
LABEL prt.vllm.commit=${VLLM_COMMIT}

# Prebuilt, or vLLM JITs them at every start. World-writable: containers run
# as the host user and still JIT a few small modules.
ENV FLASHINFER_WORKSPACE_BASE=/opt/prt/fi_ws CUDA_HOME=/usr/local/cuda
RUN CUDA_VISIBLE_DEVICES= FLASHINFER_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 \
        /opt/venv/bin/python -c 'from flashinfer.jit.comm import gen_trtllm_comm_module; from flashinfer.jit.sampling import gen_sampling_module; from flashinfer.jit.tllm_utils import gen_trtllm_utils_module; from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module; [g().build(verbose=False, need_lock=True) for g in (gen_sampling_module, gen_trtllm_comm_module, gen_trtllm_utils_module, gen_cutlass_fused_moe_sm90_module)]' \
    && chmod -R a+rwX /opt/prt/fi_ws

# uv installs without bytecode; later stages inherit UV_COMPILE_BYTECODE.
RUN /opt/venv/bin/python -m compileall -q -j 32 /opt/venv/lib/python3.12/site-packages \
    > /dev/null
ENV UV_COMPILE_BYTECODE=1

WORKDIR /workspace

#################### lmcache ####################

FROM ${SERVING_IMAGE} AS lmcache

ARG MAX_JOBS=32
ENV MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST="9.0"
COPY --from=lmcache_src . /workspace/LMCache
# Overrides, not constraints: LMCache's caps never change the serving deps.
# gRPC codegen uses LMCache's own grpcio-tools pin, kept out of the venv: the
# serving one emits code for protobuf 7, and the runtime is protobuf 6.
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv,sharing=locked \
    cd /workspace/LMCache \
    && uv pip install --python /opt/venv/bin/python \
        --overrides /opt/prt/serving-freeze.txt -r requirements/build.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
    && uv pip install --python /opt/venv/bin/python --no-deps --target /tmp/lmcache-codegen \
        $(grep -E '^grpcio(-tools)?==' requirements/build.txt) \
    && PYTHONPATH=/tmp/lmcache-codegen uv pip install --python /opt/venv/bin/python --no-build-isolation \
        --overrides /opt/prt/serving-freeze.txt . \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
    && rm -rf /workspace/LMCache /tmp/lmcache-codegen

#################### mooncake ####################
# `mooncake_wheel` context: fetched by build.sh. Its deps come from PyPI under
# the serving freeze. ldd fails the build on a missing lib; libcuda.so.1 is
# exempt, as only the NVIDIA runtime mounts it.

FROM ${SERVING_IMAGE} AS mooncake

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        libibverbs1 ibverbs-providers libnuma1 libcurl4t64 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=mooncake_wheel . /opt/prt/mooncake_wheel
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv,sharing=locked \
    uv pip install --python /opt/venv/bin/python --no-deps /opt/prt/mooncake_wheel/*.whl \
    && uv pip install --python /opt/venv/bin/python \
        -c /opt/prt/serving-freeze.txt aiohttp requests msgpack \
    && P=$(/opt/venv/bin/python -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))') \
    && test -x "$P/mooncake_master" && test -x "$P/mooncake_client" \
    && ! ldd "$P/mooncake_master" "$P/mooncake_client" | grep "not found" | grep -v libcuda.so.1

#################### aiperf ####################
# The PRT plugin, which pins AIPerf 0.13.0. --exclude-newer makes the
# resolution reproducible.

FROM python:3.12.13-slim AS aiperf

ENV PATH="/opt/venv/bin:${PATH}"
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
COPY --from=aiperf_plugins . /opt/prt/aiperf_plugins
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv-aiperf,sharing=locked \
    uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python \
        --exclude-newer 2026-10-01T00:00:00Z /opt/prt/aiperf_plugins/prt_aiperf
WORKDIR /workspace
