#!/usr/bin/env bash
# Container-side setup and repro for LMCache issue #4419.
set -euo pipefail

REPO_ROOT="/workspace/LMCache"
cd "${REPO_ROOT}"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/amd-4419-repro-results}"
LMCACHE_LOG="${RESULTS_DIR}/lmcache-server.log"
VLLM_LOG="${RESULTS_DIR}/vllm.log"
CLIENT_LOG="${RESULTS_DIR}/client.log"
CLIENT_SCRIPT="${RESULTS_DIR}/repro_lmcache_4419_client.py"
LMCACHE_PORT="${LMCACHE_PORT:-16555}"
VLLM_PORT="${VLLM_PORT:-18080}"
GPU_FOR_VLLM="${GPU_FOR_VLLM:-0}"

mkdir -p "${RESULTS_DIR}"

cleanup() {
    local status=$?
    set +e
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "${VLLM_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${LMCACHE_PID:-}" ]]; then
        kill "${LMCACHE_PID}" >/dev/null 2>&1 || true
    fi
    wait "${VLLM_PID:-}" >/dev/null 2>&1 || true
    wait "${LMCACHE_PID:-}" >/dev/null 2>&1 || true
    exit "${status}"
}
trap cleanup EXIT

echo "=== AMD #4419 repro container setup ==="
echo "repo=${REPO_ROOT}"
echo "results_dir=${RESULTS_DIR}"
echo "lmcache_port=${LMCACHE_PORT}"
echo "vllm_port=${VLLM_PORT}"
echo "gpu_for_vllm=${GPU_FOR_VLLM}"
python3 --version
uv --version
git --version || true

if ! command -v git >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends git
    rm -rf /var/lib/apt/lists/*
fi
git config --global --add safe.directory "${REPO_ROOT}"

export CXX=hipcc
export BUILD_WITH_HIP=1
export TORCH_DONT_CHECK_COMPILER_ABI=1
export SETUPTOOLS_SCM_PRETEND_VERSION="${SETUPTOOLS_SCM_PRETEND_VERSION:-0.5.4rc5.dev16}"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE:-${SETUPTOOLS_SCM_PRETEND_VERSION}}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "=== Installing build/runtime dependencies ==="
uv pip install --system --no-cache -r requirements/build.txt
uv pip uninstall --system -y cupy-cuda12x cupy-cuda11x cupy || true
uv pip install --system --no-cache "numpy<2.5" cupy-rocm-7-0
uv pip install --system --no-cache \
    "opentelemetry-exporter-prometheus<=0.61b0" \
    requests \
    sortedcontainers

# The tw22 repro image is Python 3.14 based. Use the mounted source tree plus
# in-place native extensions, matching the environment where #4419 reproduced.
echo "=== Building LMCache native extensions in-place ==="
MAX_JOBS="${MAX_JOBS:-1}" python3 setup.py build_ext --inplace

echo "=== Verifying imports and ROCm runtime ==="
vllm --help >/dev/null
python3 - <<'PY'
import cupy
import lmcache
import os
import torch
import vllm
from lmcache import device_ops
from lmcache.integration.vllm import vllm_multi_process_adapter

print(f"vLLM={vllm.__version__}")
print(f"torch={torch.__version__}, HIP={torch.version.hip}")
print(f"LMCache={lmcache.__version__}")
print(f"CuPy={cupy.__version__}, is_hip={cupy.cuda.runtime.is_hip}")
print(f"LMCache device={lmcache.torch_device_type}, ops={type(device_ops).__name__}")
print(
    "LMCache vLLM MP adapter import="
    f"{vllm_multi_process_adapter.__name__}"
)
if not cupy.cuda.runtime.is_hip:
    raise RuntimeError("Expected ROCm CuPy backend, but cupy.cuda.runtime.is_hip is false")
print(f"Selected host AMD GPU: {os.getenv('GPU_FOR_VLLM', 'unset')}")
print(f"Visible torch CUDA/HIP devices: {torch.cuda.device_count()}")
for device_idx in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(device_idx)
    total_memory_gib = props.total_memory / (1024**3)
    print(
        f"torch device {device_idx}: "
        f"name={props.name}, total_memory_gib={total_memory_gib:.1f}"
    )
PY

cat > "${CLIENT_SCRIPT}" <<'PY'
# SPDX-License-Identifier: Apache-2.0
"""Send one cold and one identical LMCache MP completion request."""

from __future__ import annotations

import json
import os
import time

import requests


BLOCK = (
    "LMCache is a KV cache layer for LLM serving. In multiprocess mode, an "
    "independent LMCache server process holds the KV cache, and vLLM connects "
    "to it as a client over ZMQ, transferring KV tensors via CUDA/HIP IPC with "
    "zero copy. This decouples cache lifetime from the inference engine and "
    "enables cache sharing. "
)
PROMPT = (
    "You are a meticulous technical assistant specializing in GPU inference "
    "systems. Background you must remember:\n\n" + BLOCK * 85
)
PAYLOAD = {
    "model": "deepseek-ai/DeepSeek-V2-Lite",
    "prompt": PROMPT,
    "max_tokens": 32,
    "temperature": 0.0,
}


def send(label: str, timeout: float) -> int:
    started = time.monotonic()
    url = f"http://localhost:{os.environ['VLLM_PORT']}/v1/completions"
    try:
        response = requests.post(url, json=PAYLOAD, timeout=timeout)
        elapsed = time.monotonic() - started
        print(f"{label}: status={response.status_code} elapsed={elapsed:.3f}s")
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: non-json response: {exc}")
            print(response.text[:2000])
            return response.status_code
        print(f"{label}: usage={body.get('usage')}")
        with open(
            f"{os.environ['RESULTS_DIR']}/response_{label}.json",
            "w",
            encoding="utf-8",
        ) as out:
            json.dump(body, out)
        return response.status_code
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - started
        print(f"{label}: ERROR elapsed={elapsed:.3f}s {type(exc).__name__}: {exc}")
        return 599


if __name__ == "__main__":
    cold = send("cold", 180.0)
    repeat = send("repeat", 30.0)
    raise SystemExit(0 if cold == 200 and repeat == 200 else 1)
PY

wait_for_vllm() {
    local deadline=$((SECONDS + 1800))
    while (( SECONDS < deadline )); do
        if curl -fsS "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
            return 0
        fi
        if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
            echo "vLLM exited before becoming healthy"
            echo "=== LMCache server log tail ==="
            tail -n 200 "${LMCACHE_LOG}" || true
            echo "=== vLLM log tail ==="
            tail -n 200 "${VLLM_LOG}" || true
            return 1
        fi
        sleep 5
    done
    echo "Timed out waiting for vLLM health endpoint"
    echo "=== LMCache server log tail ==="
    tail -n 200 "${LMCACHE_LOG}" || true
    echo "=== vLLM log tail ==="
    tail -n 200 "${VLLM_LOG}" || true
    return 1
}

wait_for_lmcache_server() {
    local deadline=$((SECONDS + 120))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${LMCACHE_PID}" >/dev/null 2>&1; then
            echo "LMCache MP server exited before accepting connections"
            cat "${LMCACHE_LOG}" || true
            return 1
        fi
        if timeout 1 bash -c "</dev/tcp/localhost/${LMCACHE_PORT}" \
            >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    echo "Timed out waiting for LMCache MP server on tcp://localhost:${LMCACHE_PORT}"
    cat "${LMCACHE_LOG}" || true
    return 1
}

echo "=== Starting LMCache MP server ==="
CUDA_VISIBLE_DEVICES="${GPU_FOR_VLLM}" \
python3 -m lmcache.cli.main server \
    --l1-size-gb 10 \
    --eviction-policy LRU \
    --port "${LMCACHE_PORT}" \
    --supported-transfer-mode lmcache_driven \
    > "${LMCACHE_LOG}" 2>&1 &
LMCACHE_PID=$!
echo "LMCache MP server PID=${LMCACHE_PID}, log=${LMCACHE_LOG}"
wait_for_lmcache_server

echo "=== Starting vLLM DeepSeek-V2-Lite with LMCache MP connector ==="
KV_TRANSFER_CONFIG="$(
    python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://localhost",
                "lmcache.mp.port": int(os.environ["LMCACHE_PORT"]),
            },
        }
    )
)
PY
)"

CUDA_VISIBLE_DEVICES="${GPU_FOR_VLLM}" \
vllm serve deepseek-ai/DeepSeek-V2-Lite \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.25 \
    --disable-hybrid-kv-cache-manager \
    --no-enable-prefix-caching \
    --kv-transfer-config "${KV_TRANSFER_CONFIG}" \
    > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
echo "vLLM PID=${VLLM_PID}, log=${VLLM_LOG}"

wait_for_vllm

echo "=== Sending cold/repeat repro requests ==="
set +e
RESULTS_DIR="${RESULTS_DIR}" VLLM_PORT="${VLLM_PORT}" \
    python3 "${CLIENT_SCRIPT}" > "${CLIENT_LOG}" 2>&1
CLIENT_STATUS=$?
set -e

sleep 5

echo "=== Client log ==="
cat "${CLIENT_LOG}" || true
echo "=== LMCache relevant log lines ==="
grep -E "Stored|Retrieved|Memory access fault|hipError|invalid argument|ERROR|Traceback" \
    "${LMCACHE_LOG}" || true
echo "=== vLLM relevant log lines ==="
grep -E "External prefix cache hit rate|EngineCore|Memory access fault|hipError|invalid argument|ERROR|Traceback" \
    "${VLLM_LOG}" || true

if [[ "${CLIENT_STATUS}" -ne 0 ]]; then
    echo "Repro requests did not both return HTTP 200."
    exit 1
fi

if ! grep -Eq "Stored [0-9]+ tokens" "${LMCACHE_LOG}"; then
    echo "LMCache server did not log a store."
    exit 1
fi

if ! grep -Eq "Retrieved [0-9]+ tokens" "${LMCACHE_LOG}"; then
    echo "Warm replay did not retrieve from LMCache."
    exit 1
fi

if grep -Eiq "Memory access fault|hipError|invalid argument|EngineDeadError" \
    "${LMCACHE_LOG}" "${VLLM_LOG}"; then
    echo "Detected ROCm/IPC failure signature in logs."
    exit 1
fi

echo "DeepSeek cold/repeat LMCache MP repro completed without reproducing #4419."
