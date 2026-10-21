#!/usr/bin/env bash
# Resolve multiprocess workloads and validate engine capabilities.

# Return code used when a valid pipeline combination is intentionally skipped
# because the selected engine does not implement the required capability.
if [[ -z "${WORKLOAD_UNSUPPORTED_STATUS+x}" ]]; then
    readonly WORKLOAD_UNSUPPORTED_STATUS=3
fi

resolve_workload_name() {
    local test_name="$1"

    case "$test_name" in
        lm_eval_preemption) printf '%s\n' lm_eval ;;
        hma_lm_eval_gemma4 | hma_lm_eval_qwen3_5) printf '%s\n' hma_lm_eval ;;
        *) printf '%s\n' "$test_name" ;;
    esac
}

engine_supports_capability() {
    local capability_array="$1"
    local requested_value="$2"
    local value
    local -a supported_values=()

    # An omitted capability array preserves compatibility with existing and
    # out-of-tree adapters by treating the capability as unrestricted.
    if ! declare -p "$capability_array" > /dev/null 2>&1; then
        return 0
    fi

    if [[ ! "$capability_array" =~ ^[A-Z_]+$ ]]; then
        echo "Invalid capability array name: $capability_array" >&2
        return 1
    fi
    eval 'supported_values=("${'"$capability_array"'[@]}")'
    for value in "${supported_values[@]}"; do
        if [[ "$value" == "$requested_value" ]]; then
            return 0
        fi
    done
    return 1
}

resolve_engine_workload() {
    local test_name="$1"
    local script_dir="$2"
    local inference_engine="$3"
    local unsupported_workload
    local workload_file_name
    local common_workload
    local engine_workload
    local transfer_mode="${LMCACHE_MP_TRANSFER_MODE:-lmcache_driven}"
    local request_transport="${LMCACHE_REQUEST_TRANSPORT:-zmq}"

    case "$transfer_mode" in
        lmcache_driven | engine_driven) ;;
        *)
            echo "Unknown LMCACHE_MP_TRANSFER_MODE='$transfer_mode'" >&2
            return 2
            ;;
    esac
    case "$request_transport" in
        zmq | grpc) ;;
        *)
            echo "Unknown LMCACHE_REQUEST_TRANSPORT='$request_transport'" >&2
            return 2
            ;;
    esac

    WORKLOAD_NAME="$(resolve_workload_name "$test_name")"
    workload_file_name="${WORKLOAD_NAME//_/-}.sh"
    common_workload="${script_dir}/workloads/common/${workload_file_name}"
    engine_workload="${script_dir}/workloads/${inference_engine}/${workload_file_name}"

    if [[ -f "$common_workload" ]]; then
        for unsupported_workload in "${ENGINE_COMMON_WORKLOAD_BLACKLIST[@]-}"; do
            if [[ "$test_name" == "$unsupported_workload" \
                    || "$WORKLOAD_NAME" == "$unsupported_workload" ]]; then
                printf "Skipping common workload '%s': %s blacklists '%s'.\n" \
                    "$test_name" "$ENGINE_NAME" "$unsupported_workload"
                return "$WORKLOAD_UNSUPPORTED_STATUS"
            fi
        done
        RESOLVED_WORKLOAD_SCRIPT="$common_workload"
    elif [[ -f "$engine_workload" ]]; then
        RESOLVED_WORKLOAD_SCRIPT="$engine_workload"
    else
        echo "Test '$test_name' is not implemented for ${ENGINE_NAME}." >&2
        echo "Expected $common_workload or $engine_workload" >&2
        return 2
    fi

    if ! engine_supports_capability \
            ENGINE_SUPPORTED_TRANSFER_MODES "$transfer_mode"; then
        printf "Skipping '%s': %s does not support transfer mode '%s'.\n" \
            "$test_name" "$ENGINE_NAME" "$transfer_mode"
        return "$WORKLOAD_UNSUPPORTED_STATUS"
    fi
    if ! engine_supports_capability \
            ENGINE_SUPPORTED_REQUEST_TRANSPORTS "$request_transport"; then
        printf "Skipping '%s': %s does not support request transport '%s'.\n" \
            "$test_name" "$ENGINE_NAME" "$request_transport"
        return "$WORKLOAD_UNSUPPORTED_STATUS"
    fi
}
