#!/usr/bin/env bash
# Validate the baked standalone image without installing into /opt/venv.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
unset PYTHONPATH
cd "${HOME}"
export PYTHONSAFEPATH=1 LMCACHE_TRACK_USAGE=false
log_dir=/workspace/build
mkdir -p "$log_dir"
exec > >(tee "${log_dir}/nonroot-standalone.log") 2>&1
id
test "$(id -un)" = lmcache
test ! -w /opt/venv
python3 -P "${SCRIPT_DIR}/nonroot_image_probe.py"
python3 -P -m lmcache.v1.multiprocess.server \
    --host 127.0.0.1 --port 5555 --l1-size-gb 0.25 \
    --no-l1-use-lazy --eviction-policy LRU > "${log_dir}/nonroot-standalone-server.log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
python3 -P - <<'PY'
import time
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

client = RequestClientFactory.create("tcp://127.0.0.1:5555")
try:
    for _ in range(30):
        try:
            ping = client.ping(None).result(timeout=3)
            chunk_size = client.get_chunk_size().result(timeout=3)
            if ping is not True or chunk_size != 256:
                raise RuntimeError(f"Unexpected RPC response: {ping}, {chunk_size}")
            print("NONROOT_STANDALONE_RPC_PASS", flush=True)
            break
        except TimeoutError:
            time.sleep(1)
    else:
        raise RuntimeError("Standalone server did not respond")
finally:
    client.close()
PY
