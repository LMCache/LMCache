#!/usr/bin/env bash
# Runs a GPU test command while github_ci_gate.py watches the PR's GitHub
# Layer 2 checks (test.yml, cpu_device.yml) in parallel.
#   Layer 2 fails or is cancelled -> stop the test command, fail the step and
#                                    release the GPU early.
#   Layer 2 passes                -> the test command runs on to its own result.
# Non-PR builds just run the command.
#
# Usage: run-gpu-with-l2-watch.sh <command> [args...]
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# A new session gives the tests their own process group, so one signal stops
# pytest and its workers. Without job control, a background child is not a
# process group leader, so setsid does not fork and $! is the session leader.
setsid "$@" &
test_pid=$!
python3 "${SCRIPT_DIR}/github_ci_gate.py" watch-l2 &
watch_pid=$!

while kill -0 "${test_pid}" 2>/dev/null; do
    if ! kill -0 "${watch_pid}" 2>/dev/null; then
        if wait "${watch_pid}"; then
            wait "${test_pid}"
            exit $?
        fi
        echo "+++ :octagonal_sign: GitHub Layer 2 failed; stopping the GPU tests"
        kill -TERM -- "-${test_pid}" 2>/dev/null || kill -TERM "${test_pid}"
        wait "${test_pid}"
        exit 1
    fi
    sleep 2
done

kill "${watch_pid}" 2>/dev/null
wait "${test_pid}"
