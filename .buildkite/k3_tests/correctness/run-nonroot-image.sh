#!/usr/bin/env bash
# Exercise the baked vLLM image with prefix caching disabled.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
unset PYTHONPATH
cd "${HOME}"
export PYTHONSAFEPATH=1 LMCACHE_TRACK_USAGE=false PYTHONHASHSEED=0
log_dir=/workspace/build
mkdir -p "$log_dir"
exec > >(tee "${log_dir}/nonroot-vllm.log") 2>&1
id
test "$(id -un)" = lmcache
test ! -w /opt/venv
python3 -P "${SCRIPT_DIR}/../unit/nonroot_image_probe.py"
run_dir="$(mktemp -d "${HOME}/nonroot-e2e.XXXXXX")"
cat > "${run_dir}/lmcache.yaml" <<EOF_CONFIG
chunk_size: 256
local_cpu: true
max_local_cpu_size: 1
local_disk: "file://${run_dir}/disk"
max_local_disk_size: 1
extra_config:
  enable_cache_usage_details_in_response: true
EOF_CONFIG
export LMCACHE_CONFIG_FILE="${run_dir}/lmcache.yaml"
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --host 127.0.0.1 --port 8000 --max-model-len 2048 \
    --gpu-memory-utilization 0.3 --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    > "${log_dir}/nonroot-vllm-server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true; rm -rf "$run_dir"' EXIT
python3 -P "${SCRIPT_DIR}/nonroot_image_client.py"
test -n "$(find "${run_dir}/disk" -type f -print -quit)"
echo NONROOT_LOCAL_DISK_WRITE_PASS
