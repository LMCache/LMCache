#!/usr/bin/env bash
# vLLM adapter for the LMCache multiprocess integration-test harness.

export ENGINE_NAME="vLLM"
export ENGINE_DEFAULT_MODEL="Qwen/Qwen3-14B"

ENGINE_SUPPORTED_TRANSFER_MODES=(lmcache_driven engine_driven)
ENGINE_SUPPORTED_REQUEST_TRANSPORTS=(zmq grpc)

# Common workloads that this adapter cannot currently run. Entries use the
# normalized workload name with underscores, for example: long_doc_qa_l2.
ENGINE_COMMON_WORKLOAD_BLACKLIST=()

engine_setup_environment() {
    local repo_root="$1"
    local setup_script="${BK_SETUP_ENV_SCRIPT:-${repo_root}/.buildkite/k3_harness/setup-env.sh}"
    export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-${TORCH_DEVICE_TYPE:-cuda}}"
    export GPU_MEMORY_PROBE_ENABLED="${GPU_MEMORY_PROBE_ENABLED:-1}"
    export BATCH_INVARIANT_DEFAULT="${BATCH_INVARIANT_DEFAULT:-1}"
    source "$setup_script"
}

engine_configure_defaults() {
    export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-${TORCH_DEVICE_TYPE:-cuda}}"
    export DEVICE_AFFINITY_VAR="${DEVICE_AFFINITY_VAR:-CUDA_VISIBLE_DEVICES}"
    export GPU_MEMORY_PROBE_ENABLED="${GPU_MEMORY_PROBE_ENABLED:-1}"
    export BATCH_INVARIANT_DEFAULT="${BATCH_INVARIANT_DEFAULT:-1}"

    export ENGINE_PORT="${ENGINE_PORT:-${VLLM_PORT:-8000}}"
    export ENGINE_BASELINE_PORT="${ENGINE_BASELINE_PORT:-${VLLM_BASELINE_PORT:-9000}}"
    export GPU_FOR_ENGINE="${GPU_FOR_ENGINE:-${GPU_FOR_VLLM:-0}}"
    export GPU_FOR_BASELINE="${GPU_FOR_BASELINE:-1}"

    # Keep the established variables available to vLLM-specific MP tests.
    export VLLM_PORT="$ENGINE_PORT"
    export VLLM_BASELINE_PORT="$ENGINE_BASELINE_PORT"
    export GPU_FOR_VLLM="$GPU_FOR_ENGINE"

    export ENGINE_LOG_FILE="${ENGINE_LOG_FILE:-/tmp/build_${BUILD_ID:-local_$$}_vllm.log}"
    export ENGINE_BASELINE_LOG_FILE="${ENGINE_BASELINE_LOG_FILE:-/tmp/build_${BUILD_ID:-local_$$}_vllm_baseline.log}"
}

engine_prepare_launch() {
    local device_index="$1"
    local gpu_memory_gb=0

    VLLM_GPU_MEMORY_ARGS=()
    if [[ "$GPU_MEMORY_PROBE_ENABLED" == "1" || "$GPU_MEMORY_PROBE_ENABLED" == "true" ]]; then
        local gpu_memory_mb
        gpu_memory_mb=$(
            env "${DEVICE_AFFINITY_VAR}=${device_index}" \
                python3 - <<'PY'
from lmcache import torch_dev

print(torch_dev.get_device_properties(0).total_memory // (1024 * 1024))
PY
        )
        gpu_memory_gb=$((gpu_memory_mb / 1024))
        echo "Detected GPU memory: ${gpu_memory_gb}GB (${gpu_memory_mb}MB)"
    else
        echo "GPU memory probe disabled"
    fi

    if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
        echo "Using configured --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}"
        VLLM_GPU_MEMORY_ARGS=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
    elif ((gpu_memory_gb > 90)); then
        echo "GPU memory > 90GB, adding --gpu-memory-utilization 0.5"
        VLLM_GPU_MEMORY_ARGS=(--gpu-memory-utilization 0.5)
    fi

    VLLM_ATTENTION_ARGS=()
    if [[ -n "${ATTENTION_BACKEND:-FLASH_ATTN}" && "${ATTENTION_BACKEND:-FLASH_ATTN}" != "auto" ]]; then
        VLLM_ATTENTION_ARGS=(--attention-backend "${ATTENTION_BACKEND:-FLASH_ATTN}")
    fi

    VLLM_ATTENTION_ENV=()
    if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
        echo "Using VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}"
        VLLM_ATTENTION_ENV=("VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND}")
    fi

    VLLM_EAGER_ARGS=()
    if [[ "${ENFORCE_EAGER:-0}" == "1" || "${ENFORCE_EAGER:-0}" == "true" ]]; then
        VLLM_EAGER_ARGS=(--enforce-eager)
    fi

    VLLM_PREFIX_ARGS=()
    VLLM_MAMBA_ARGS=()
    if [[ "${VLLM_DISABLE_PREFIX_CACHING:-false}" == "1" || "${VLLM_DISABLE_PREFIX_CACHING:-false}" == "true" ]]; then
        echo "Disabling vLLM prefix caching via --no-enable-prefix-caching"
        VLLM_PREFIX_ARGS=(--no-enable-prefix-caching)
    elif [[ -n "${MAMBA_CACHE_MODE:-}" ]]; then
        VLLM_MAMBA_ARGS=(--mamba-cache-mode "$MAMBA_CACHE_MODE")
        VLLM_PREFIX_ARGS=(--enable-prefix-caching)
    fi

    VLLM_BATCH_ARGS=()
    if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
        VLLM_BATCH_ARGS=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
    fi

    # Pin the KV block pool (in blocks) on both servers. The preemption test
    # needs the pool size to be independent of the GPU model: a workload whose
    # KV demand exceeds NUM_GPU_BLOCKS_OVERRIDE * block_size preempts by
    # arithmetic. Empty -> vLLM sizes the pool from --gpu-memory-utilization.
    VLLM_POOL_ARGS=()
    if [[ -n "${NUM_GPU_BLOCKS_OVERRIDE:-}" ]]; then
        VLLM_POOL_ARGS=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS_OVERRIDE")
    fi

    # Async scheduling on both servers. Off by default: the determinism tests
    # were qualified with it off. With a consumer-role connector vLLM then
    # defers block frees to the end of the in-flight step, which is a
    # different preemption path, so the preemption matrix runs both.
    VLLM_SCHEDULING_ARGS=(--no-async-scheduling)
    if [[ "${ASYNC_SCHEDULING:-0}" == "1" || "${ASYNC_SCHEDULING:-0}" == "true" ]]; then
        VLLM_SCHEDULING_ARGS=(--async-scheduling)
    fi
}

engine_add_lmcache_server_environment() {
    local environment_name="$1"
    local -n environment="$environment_name"
    environment+=("VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}")
}

engine_kv_transfer_config() {
    LMCACHE_PORT="${LMCACHE_PORT}" \
    LMCACHE_REQUEST_SCHEME="${LMCACHE_REQUEST_SCHEME}" \
    LMCACHE_MP_MQ_TIMEOUT="${LMCACHE_MP_MQ_TIMEOUT:-10}" \
    LMCACHE_MP_LAZY_OFFLOAD="${LMCACHE_MP_LAZY_OFFLOAD:-false}" \
    LMCACHE_MP_LAZY_OFFLOAD_THRESHOLD="${LMCACHE_MP_LAZY_OFFLOAD_THRESHOLD:-2}" \
    LMCACHE_MP_LAZY_OFFLOAD_SELECT_COUNT="${LMCACHE_MP_LAZY_OFFLOAD_SELECT_COUNT:-1}" \
        python3 - <<'PY'
import json
import os

extra_config = {
    "lmcache.mp.host": os.environ["LMCACHE_REQUEST_SCHEME"] + "://localhost",
    "lmcache.mp.port": int(os.environ["LMCACHE_PORT"]),
    "lmcache.mp.mq_timeout": int(os.environ["LMCACHE_MP_MQ_TIMEOUT"]),
}
if os.environ["LMCACHE_MP_LAZY_OFFLOAD"].lower() in {"1", "true"}:
    extra_config.update(
        {
            "lmcache.mp.lazy_offload": True,
            "lmcache.mp.lazy_offload_policy": "FIFO",
            "lmcache.mp.lazy_offload_threshold": int(
                os.environ["LMCACHE_MP_LAZY_OFFLOAD_THRESHOLD"]
            ),
            "lmcache.mp.lazy_offload_select_count": int(
                os.environ["LMCACHE_MP_LAZY_OFFLOAD_SELECT_COUNT"]
            ),
        }
    )

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_load_failure_policy": "recompute",
            "kv_connector_extra_config": extra_config,
        }
    )
)
PY
}

engine_launch() {
    local mode="$1"
    local port="$2"
    local device_index="$3"
    local log_file="$4"
    local batch_invariant="${BATCH_INVARIANT:-${BATCH_INVARIANT_DEFAULT}}"
    local -a mode_args=()

    if [[ "$mode" == "lmcache" ]]; then
        local kv_transfer_config
        kv_transfer_config="$(engine_kv_transfer_config)"
        echo "LMCache KV transfer configuration: ${kv_transfer_config}"
        mode_args=(
            --kv-transfer-config "$kv_transfer_config"
            "${VLLM_MAMBA_ARGS[@]}"
            "${VLLM_BATCH_ARGS[@]}"
        )
    elif [[ "$mode" != "baseline" ]]; then
        echo "Unknown ${ENGINE_NAME} launch mode: $mode" >&2
        return 1
    fi

    env -u VLLM_PORT \
        "${DEVICE_AFFINITY_VAR}=${device_index}" \
        "VLLM_TARGET_DEVICE=${VLLM_TARGET_DEVICE}" \
        "${VLLM_ATTENTION_ENV[@]}" \
        VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        VLLM_SERVER_DEV_MODE=1 \
        "VLLM_BATCH_INVARIANT=${batch_invariant}" \
        PYTHONHASHSEED=0 \
        vllm serve "$MODEL" \
            "${mode_args[@]}" \
            "${VLLM_ATTENTION_ARGS[@]}" \
            --port "$port" \
            "${VLLM_SCHEDULING_ARGS[@]}" \
            --max-model-len "${MAX_MODEL_LEN:-auto}" \
            "${VLLM_EAGER_ARGS[@]}" \
            "${VLLM_GPU_MEMORY_ARGS[@]}" \
            "${VLLM_POOL_ARGS[@]}" \
            "${VLLM_PREFIX_ARGS[@]}" \
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
        deadlock)
            if [[ "$mode" != "lmcache" ]]; then
                echo "The deadlock profile requires LMCache mode" >&2
                return 1
            fi
            _engine_launch_deadlock_profile "$port" "$log_file"
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
    printf 'http://127.0.0.1:%s/v1/models\n' "$port"
}

engine_clear_local_cache() {
    local port="$1"
    curl --noproxy '*' -fsS --max-time 60 -X POST \
        "http://127.0.0.1:${port}/reset_prefix_cache" > /dev/null
}

engine_count_preemptions() {
    local log_file="$1"
    if [[ ! -f "$log_file" ]]; then
        echo 0
        return 0
    fi
    local count
    count=$(grep -c "<preempted>" "$log_file" 2>/dev/null || true)
    echo "${count:-0}"
}

engine_print_timeout_diagnostics() {
    local log_file="$1"
    local role pid

    if [[ ! -f "$log_file" ]]; then
        return 0
    fi

    for role in EngineCore APIServer; do
        pid=$(grep -oE "\\(${role} pid=[0-9]+\\)" "$log_file" \
            | tail -n 1 | sed -E 's/.*pid=([0-9]+).*/\1/' || true)
        if [[ -n "$pid" ]] && declare -F print_process_diagnostics >/dev/null; then
            print_process_diagnostics "$pid" "$role"
        else
            echo "${role} PID not found in $log_file"
        fi
    done
}

engine_cleanup_processes() {
    return 0
}

_engine_launch_deadlock_profile() {
    local port="$1"
    local log_file="$2"
    local kv_transfer_config

    kv_transfer_config="$(
        LMCACHE_MP_MQ_TIMEOUT=60 engine_kv_transfer_config
    )"
    env -u VLLM_PORT \
        FLASHINFER_DISABLE_VERSION_CHECK=1 \
        VLLM_SERVER_DEV_MODE=1 \
        vllm serve "$MODEL" \
            --tensor-parallel-size 2 \
            --distributed-executor-backend mp \
            --block-size 64 \
            --trust-remote-code \
            --load-format dummy \
            --enable-prefix-caching \
            --enable-chunked-prefill \
            --gpu-memory-utilization 0.8 \
            --max-model-len 65536 \
            --hf-overrides '{"max_position_embeddings":65536}' \
            --max-num-seqs 32 \
            --max-num-batched-tokens 16000 \
            --scheduling-policy fcfs \
            --port "$port" \
            --enforce-eager \
            --kv-transfer-config "$kv_transfer_config" \
            > "$log_file" 2>&1 &
    ENGINE_PID=$!
}
