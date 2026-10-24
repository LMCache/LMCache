# PRT images. Build with build.sh.
#
#   serving  the upstream vLLM nightly image of VLLM_COMMIT, no source
#            patches; adds only startup packaging. Shared by every deployment.
#   lmcache  serving + the LMCache under test (`lmcache_src` context); runs
#            both `vllm serve` and `lmcache server`.
#   mooncake serving + the Mooncake nightly wheel; also runs mooncake_master
#            and mooncake_client.
#   aiperf   the load generator.

# build.sh passes vllm/vllm-openai:nightly-<VLLM_COMMIT>.
ARG VLLM_IMAGE=vllm/vllm-openai:nightly
# Base of the lmcache and mooncake stages; build.sh passes the pinned image.
ARG SERVING_IMAGE=serving

#################### serving ####################
# The runner passes the whole command, so the image's `vllm serve` entrypoint
# goes. The LMCache and Mooncake it bundles go too: the lmcache and mooncake
# stages install the only copy, at the version under test.

FROM ${VLLM_IMAGE} AS serving

ENTRYPOINT []
# The image points uv's cache into its own layers; later stages mount it.
ENV UV_CACHE_DIR=/root/.cache/uv
ARG VLLM_COMMIT
# The freeze drops `name @ file://` lines: those wheels are gone from the image.
RUN uv pip uninstall --system lmcache mooncake-transfer-engine-cuda13 \
    && mkdir -p /opt/prt \
    && uv pip freeze --system | grep -v ' @ ' > /opt/prt/serving-freeze.txt
LABEL prt.vllm.commit=${VLLM_COMMIT}

# Prebuilt, or vLLM JITs them at every start. World-writable: containers run
# as the host user and still JIT a few small modules.
ENV FLASHINFER_WORKSPACE_BASE=/opt/prt/fi_ws
RUN mkdir -p /opt/prt/fi_ws \
    && CUDA_VISIBLE_DEVICES= FLASHINFER_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 \
        python3 -c 'from flashinfer.jit.comm import gen_trtllm_comm_module; from flashinfer.jit.sampling import gen_sampling_module; from flashinfer.jit.tllm_utils import gen_trtllm_utils_module; from flashinfer.jit.fused_moe import gen_cutlass_fused_moe_sm90_module; [g().build(verbose=False, need_lock=True) for g in (gen_sampling_module, gen_trtllm_comm_module, gen_trtllm_utils_module, gen_cutlass_fused_moe_sm90_module)]' \
    && chmod -R a+rwX /opt/prt/fi_ws

WORKDIR /workspace

#################### lmcache ####################

FROM ${SERVING_IMAGE} AS lmcache

# The vLLM image lacks what building LMCache needs: git for setuptools-scm's
# version, and the cuSPARSE and cuSOLVER headers that ATen/cuda/CUDAContext.h
# includes.
RUN CUDA_VERSION_DASH=$(echo $CUDA_VERSION | cut -d. -f1,2 | tr '.' '-') \
    && apt-get update -y \
    && apt-get install -y --no-install-recommends git \
        libcusparse-dev-${CUDA_VERSION_DASH} libcusolver-dev-${CUDA_VERSION_DASH} \
    && rm -rf /var/lib/apt/lists/*
# The DP router of vLLM's recipes; its Python deps are already in the image.
RUN uv pip install --system --no-deps vllm-router==0.1.16
ARG MAX_JOBS=32
ENV MAX_JOBS=${MAX_JOBS} TORCH_CUDA_ARCH_LIST="9.0"
COPY --from=lmcache_src . /workspace/LMCache
# Overrides, not constraints: LMCache's caps never change the serving deps.
# gRPC codegen uses LMCache's own grpcio-tools pin, kept out of the image's
# packages: a newer one emits code for protobuf 7, and the runtime is protobuf 6.
RUN --mount=type=cache,target=/root/.cache/uv,id=prt-uv,sharing=locked \
    cd /workspace/LMCache \
    && uv pip install --system \
        --overrides /opt/prt/serving-freeze.txt -r requirements/build.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --index-strategy unsafe-best-match \
    && uv pip install --system --no-deps --target /tmp/lmcache-codegen \
        $(grep -E '^grpcio(-tools)?==' requirements/build.txt) \
    && PYTHONPATH=/tmp/lmcache-codegen uv pip install --system --no-build-isolation \
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
    uv pip install --system --no-deps /opt/prt/mooncake_wheel/*.whl \
    && uv pip install --system \
        -c /opt/prt/serving-freeze.txt aiohttp requests msgpack \
    && P=$(python3 -c 'import mooncake, os; print(os.path.dirname(mooncake.__file__))') \
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
