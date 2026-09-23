#!/usr/bin/env bash
# SGLang adapter for the LMCache multiprocess integration-test harness.

export ENGINE_NAME="SGLang"
export ENGINE_DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"

ENGINE_SUPPORTED_TRANSFER_MODES=(lmcache_driven)
ENGINE_SUPPORTED_REQUEST_TRANSPORTS=(zmq grpc)

# These common workloads need engine behavior that the unified SGLang
# integration does not expose yet. Keep the exceptions explicit so a manual
# invocation skips before launching servers.
ENGINE_COMMON_WORKLOAD_BLACKLIST=(
    deadlock
    lm_eval_preemption
    long_doc_qa_l2
    restart_recovery
)

engine_setup_environment() {
    local repo_root="$1"
    local setup_script="${BK_SETUP_ENV_SCRIPT:-${repo_root}/.buildkite/k3_harness/setup-sglang-env.sh}"

    # Remove this default once the SGLang integration lands upstream. Callers
    # can pin a commit or wheel by overriding SGLANG_INSTALL_SPEC.
    export SGLANG_INSTALL_SPEC="${SGLANG_INSTALL_SPEC:-git+https://github.com/chunxiaozheng/sglang.git@lmcache/lmcache-unified-radix-cache#subdirectory=python}"
    source "$setup_script"
}

engine_configure_defaults() {
    export DEVICE_AFFINITY_VAR="${DEVICE_AFFINITY_VAR:-CUDA_VISIBLE_DEVICES}"
    export ENGINE_PORT="${ENGINE_PORT:-${SGLANG_PORT:-8000}}"
    export ENGINE_BASELINE_PORT="${ENGINE_BASELINE_PORT:-${SGLANG_BASELINE_PORT:-9000}}"
    export GPU_FOR_ENGINE="${GPU_FOR_ENGINE:-${GPU_FOR_SGLANG:-0}}"
    export GPU_FOR_BASELINE="${GPU_FOR_BASELINE:-1}"

    # Keep engine-specific aliases available to diagnostics and local callers.
    export SGLANG_PORT="$ENGINE_PORT"
    export SGLANG_BASELINE_PORT="$ENGINE_BASELINE_PORT"
    export GPU_FOR_SGLANG="$GPU_FOR_ENGINE"

    export ENGINE_LOG_FILE="${ENGINE_LOG_FILE:-/tmp/build_${BUILD_ID:-local_$$}_sglang.log}"
    export ENGINE_BASELINE_LOG_FILE="${ENGINE_BASELINE_LOG_FILE:-/tmp/build_${BUILD_ID:-local_$$}_sglang_baseline.log}"
    export SGLANG_LMCACHE_CONFIG_FILE="${SGLANG_LMCACHE_CONFIG_FILE:-/tmp/lmcache_sglang_${BUILD_ID:-local_$$}.yaml}"

    if [[ "${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}" != "lmcache_driven" ]]; then
        echo "SGLang unified LMCache integration supports only lmcache_driven mode" >&2
        return 1
    fi
}

engine_configure_workload() {
    local test_name="$1"

    case "$test_name" in
        long_doc_qa)
            # vLLM's stricter defaults are calibrated against its own
            # baseline. Initially require SGLang to improve TTFT while keeping
            # total query-round time within a bounded regression.
            export MAX_TTFT_SLOWDOWN_PCT="${MAX_TTFT_SLOWDOWN_PCT:-0}"
            export MAX_ROUND_TIME_SLOWDOWN_PCT="${MAX_ROUND_TIME_SLOWDOWN_PCT:-20}"
            ;;
        lm_eval)
            # Clear SGLang's radix cache between runs so the second pass proves
            # that LMCache, rather than only the local radix tree, served KV.
            export VERIFY_LMCACHE_RETRIEVAL="${VERIFY_LMCACHE_RETRIEVAL:-true}"
            ;;
    esac
}

engine_prepare_launch() {
    local device_index="$1"
    : "$device_index"

    SGLANG_MEMORY_ARGS=()
    if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
        SGLANG_MEMORY_ARGS=(--mem-fraction-static "$GPU_MEMORY_UTILIZATION")
    elif [[ -n "${SGLANG_MEM_FRACTION_STATIC:-}" ]]; then
        SGLANG_MEMORY_ARGS=(--mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC")
    fi

    SGLANG_CONTEXT_ARGS=()
    if [[ -n "${MAX_MODEL_LEN:-}" && "${MAX_MODEL_LEN}" != "auto" ]]; then
        SGLANG_CONTEXT_ARGS=(--context-length "$MAX_MODEL_LEN")
    fi
}

engine_add_lmcache_server_environment() {
    local environment_name="$1"
    # The server and SGLang workers share the selected CUDA device. No
    # SGLang-specific variables are required by the LMCache process.
    : "$environment_name"
}

engine_write_lmcache_config() {
    cat > "$SGLANG_LMCACHE_CONFIG_FILE" <<EOF
mp_host: ${LMCACHE_REQUEST_SCHEME}://127.0.0.1
mp_port: ${LMCACHE_PORT}
extra_config:
  lmcache.mp.mq_timeout: ${LMCACHE_MP_MQ_TIMEOUT:-60}
EOF
}

engine_launch() {
    local mode="$1"
    local port="$2"
    local device_index="$3"
    local log_file="$4"
    local -a mode_args=()

    if [[ "$mode" == "lmcache" ]]; then
        engine_write_lmcache_config
        mode_args=(
            --enable-lmcache
            --lmcache-config-file "$SGLANG_LMCACHE_CONFIG_FILE"
        )
    elif [[ "$mode" != "baseline" ]]; then
        echo "Unknown ${ENGINE_NAME} launch mode: $mode" >&2
        return 1
    fi

    env "${DEVICE_AFFINITY_VAR}=${device_index}" \
        PYTHONHASHSEED=0 \
        python3 -m sglang.launch_server \
            --model-path "$MODEL" \
            --host 127.0.0.1 \
            --port "$port" \
            "${mode_args[@]}" \
            "${SGLANG_MEMORY_ARGS[@]}" \
            "${SGLANG_CONTEXT_ARGS[@]}" \
            > "$log_file" 2>&1 &
    ENGINE_PID=$!
}

engine_launch_profile() {
    local profile="$1"
    local mode="$2"
    local port="$3"
    local device_index="$4"
    local log_file="$5"

    case "$profile" in
        default)
            engine_prepare_launch "$device_index"
            engine_launch "$mode" "$port" "$device_index" "$log_file"
            ;;
        *)
            echo "Unsupported ${ENGINE_NAME} launch profile: $profile" >&2
            return 1
            ;;
    esac
}

engine_ready_urls() {
    local port="$1"
    printf 'http://127.0.0.1:%s/health\n' "$port"
}

engine_clear_local_cache() {
    local port="$1"
    curl --noproxy '*' -fsS --max-time 60 -X POST \
        "http://127.0.0.1:${port}/flush_cache?timeout=60" > /dev/null
}

engine_cleanup_processes() {
    pkill -TERM -f "sglang::scheduler" 2>/dev/null || true
    pkill -TERM -f "sglang.srt" 2>/dev/null || true
}
