#!/usr/bin/env bash
# Shared helper functions for K3s test scripts.
# Source this file from every scripts/*.sh.

# Track background PIDs for cleanup
TRACKED_PIDS=()

github_https_remote_url() {
    local remote_url="$1"
    case "$remote_url" in
        https://*|http://*)
            printf '%s\n' "$remote_url"
            ;;
        git@github.com:*)
            printf 'https://github.com/%s\n' "${remote_url#git@github.com:}"
            ;;
        ssh://git@github.com/*)
            printf 'https://github.com/%s\n' "${remote_url#ssh://git@github.com/}"
            ;;
        *)
            return 1
            ;;
    esac
}

# Merge the PR base branch into the current checkout for Buildkite PR builds.
# Buildkite's checkout step may leave the pod on the raw PR head commit rather
# than a synthetic merge commit, so tests can miss conflicts/regressions that
# only appear once the latest base branch is merged in.
merge_pr_base_branch() {
    local pr_number="${BUILDKITE_PULL_REQUEST:-false}"
    if [[ -z "${pr_number}" || "${pr_number}" == "false" ]]; then
        echo "--- :git: Not a PR build; skipping base-branch pre-merge"
        return 0
    fi

    local base_branch="${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-dev}"
    local remote_name="${1:-origin}"
    local merge_ref="refs/remotes/${remote_name}/${base_branch}"
    local fetch_source="${remote_name}"
    local remote_url
    remote_url="$(git remote get-url "${remote_name}")"

    # The checkout phase may have SSH available even when the test container
    # does not. Convert GitHub SSH remotes to HTTPS so the pre-merge fetch is
    # portable across the different K3 job images.
    if fetch_source="$(github_https_remote_url "${remote_url}")"; then
        :
    else
        fetch_source="${remote_name}"
    fi

    echo "--- :git: Pre-merging ${remote_name}/${base_branch} into PR head"

    git fetch --no-tags "${fetch_source}" \
        "+refs/heads/${base_branch}:${merge_ref}"

    if git merge-base --is-ancestor "${merge_ref}" HEAD; then
        echo "  Current checkout already contains ${remote_name}/${base_branch}"
        return 0
    fi

    local git_user_name
    git_user_name="$(git config --get user.name || true)"
    if [[ -z "${git_user_name}" ]]; then
        git config user.name "buildkite-ci"
    fi

    local git_user_email
    git_user_email="$(git config --get user.email || true)"
    if [[ -z "${git_user_email}" ]]; then
        git config user.email "buildkite-ci@lmcache.ai"
    fi

    GIT_MERGE_AUTOEDIT=no git merge --no-ff --no-edit "${merge_ref}"
}

# Check that all visible GPUs have sufficient free memory.
# Fails fast with a clear message if a host-level process is hogging GPU memory.
# Usage: check_gpu_health [min_free_percent]
#   min_free_percent: minimum percentage of GPU memory that must be free (default: 80)
check_gpu_health() {
    local min_free_pct="${1:-80}"
    echo "--- :mag: GPU health check (require ${min_free_pct}% free memory)"

    if ! command -v nvidia-smi &>/dev/null; then
        echo "  nvidia-smi not found, skipping GPU health check"
        return 0
    fi

    # Parse GPU info into a temp file to avoid subshell variable scoping issues
    local gpu_info
    gpu_info=$(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null | sed 's/ //g')

    if [[ -z "$gpu_info" ]]; then
        echo "  No GPUs detected, skipping"
        return 0
    fi

    local has_problem=false
    while IFS=, read -r idx total used free; do

        if [[ "$total" -eq 0 ]]; then
            continue
        fi

        local free_pct=$((free * 100 / total))

        if [[ "$free_pct" -lt "$min_free_pct" ]]; then
            echo "  WARNING: GPU $idx has only ${free_pct}% free (${free} MiB free / ${total} MiB total, ${used} MiB used)"
            # Show which processes are using this GPU
            nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader -i "$idx" 2>/dev/null | while IFS= read -r proc; do
                echo "    Process: $proc"
            done
            has_problem=true
        else
            echo "  GPU $idx: OK (${free_pct}% free, ${free} MiB / ${total} MiB)"
        fi
    done <<< "$gpu_info"

    if [[ "$has_problem" == "true" ]]; then
        echo ""
        echo "FATAL: One or more GPUs have insufficient free memory."
        echo "This usually means a stale process on the host is consuming GPU memory."
        echo "Check the host with: nvidia-smi"
        echo "To fix: kill the offending host processes, then re-run the CI job."
        return 1
    fi

    echo "  All GPUs healthy."
    return 0
}

# Find an available TCP port starting from a given port number.
# Usage: find_free_port [start_port]
find_free_port() {
    local port="${1:-8000}"
    while [ "$port" -lt 65536 ]; do
        if ! lsof -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 &&
           ! timeout 1 bash -c "</dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
            echo "$port"
            return 0
        fi
        ((port++))
    done
    echo "ERROR: No available port found starting from ${1:-8000}" >&2
    return 1
}

# Wait for a vLLM server to become ready by polling /v1/models.
# Usage: wait_for_server <port> [timeout_secs] [log_file]
# If log_file is provided, its tail is dumped to stderr on timeout so the
# real failure (e.g. an ImportError during startup) is visible inline in the
# job output instead of requiring a trip through build artifacts.
wait_for_server() {
    local port="$1"
    local timeout="${2:-180}"
    local log_file="${3:-}"
    echo "Waiting for vLLM on port $port (timeout=${timeout}s)..."
    for ((i = 0; i < timeout; i++)); do
        if curl -sf "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            echo "vLLM ready on port $port (${i}s)"
            return 0
        fi
        sleep 1
    done
    echo "vLLM failed to start on port $port within ${timeout}s" >&2
    if [[ -n "$log_file" && -f "$log_file" ]]; then
        echo "--- :page_facing_up: Last 200 lines of ${log_file}" >&2
        tail -n 200 "$log_file" >&2
    fi
    return 1
}

# Kill all tracked background PIDs and wait for them.
# Call this in a trap handler.
cleanup_pids() {
    echo "--- Cleaning up background processes..."
    for pid in "${TRACKED_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    TRACKED_PIDS=()
    # Give GPU memory a moment to release
    sleep 2
}
