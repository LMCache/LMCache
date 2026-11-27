#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../../.."
source "${SCRIPT_DIR}/../common_scripts/path-filter.sh"

# Run the suite if change detection fails; never skip on an unknown diff.
if changed_files=$(_path_filter_get_changed_files) && [[ -n "${changed_files}" ]]; then
    relevant=false
    while IFS= read -r file; do
        case "${file}" in
            lmcache/integration/vllm/experimental/token_drop_worker.py|\
            lmcache/integration/vllm/experimental/token_drop_kv.py|\
            tests/v1/test_token_drop_worker.py|\
            tests/v1/test_token_drop_gpu_views.py|\
            .buildkite/k3_tests/token-dropping/*)
                relevant=true
                break
                ;;
            lmcache/integration/vllm/experimental/token_drop.py|\
            lmcache/integration/vllm/experimental/token_drop_adapter.py)
                # These are GPU worker dependencies once the worker exists.
                if [[ -f lmcache/integration/vllm/experimental/token_drop_worker.py ]]; then
                    relevant=true
                    break
                fi
                ;;
        esac
    done <<< "${changed_files}"
    if [[ "${relevant}" == false ]]; then
        echo "--- No token-dropping GPU changes; skipping GPU allocation"
        exit 0
    fi
fi

exec buildkite-agent pipeline upload "${SCRIPT_DIR}/pipeline.yml"
