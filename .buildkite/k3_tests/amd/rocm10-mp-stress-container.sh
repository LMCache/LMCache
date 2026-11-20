#!/usr/bin/env bash
# Runs inside the ROCm 10 vLLM image: build LMCache, start the MP server and
# vLLM at TP=8 (every rank writes every step), then hammer it with stores and
# retrieves for STRESS_MINUTES. Fails if the engine dies, the bench fails, or
# either log shows a device error.
set -uo pipefail
cd /workspace/LMCache
export CXX=hipcc BUILD_WITH_HIP=1 TORCH_DONT_CHECK_COMPILER_ABI=1
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE="0.0.0+ci"
MODEL="${MODEL:-Qwen/Qwen3-14B}"
TP="${TENSOR_PARALLEL_SIZE:-8}"
STRESS_MINUTES="${STRESS_MINUTES:-20}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
NUM_PROMPTS="${NUM_PROMPTS:-2000}"
OUT=/workspace/LMCache/rocm10-stress-results
mkdir -p "$OUT"

uv pip install --system --no-cache -r requirements/build.txt
uv pip install --system --no-cache --no-build-isolation -e . || { echo "LMCache build failed"; exit 2; }
python3 lmcache/v1/multiprocess/transport/grpc_impl/_proto_gen/_generate.py

python3 -m lmcache.v1.multiprocess.http_server \
    --l1-size-gb "${CPU_BUFFER_SIZE:-80}" --eviction-policy LRU \
    --max-workers 8 --max-gpu-workers 8 --transport zmq \
    --port 6555 --http-port 6556 --chunk-size "${CHUNK_SIZE:-256}" \
    --supported-transfer-mode lmcache_driven > "$OUT/server.log" 2>&1 &
SERVER=$!
sleep 15

VLLM_SERVER_DEV_MODE=1 vllm serve "$MODEL" --tensor-parallel-size "$TP" --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.host":"tcp://localhost","lmcache.mp.port":6555,"lmcache.mp.mq_timeout":10}}' \
    > "$OUT/vllm.log" 2>&1 &
VLLM=$!
for _ in $(seq 1 180); do
    curl -sf http://localhost:8000/health >/dev/null 2>&1 && break
    kill -0 "$VLLM" 2>/dev/null || { echo "vLLM exited during startup"; tail -40 "$OUT/vllm.log"; exit 1; }
    sleep 5
done
curl -sf http://localhost:8000/health >/dev/null || { echo "vLLM never became healthy"; exit 1; }

# Pass with seed 0 stores; the same seed again retrieves. Alternate until the
# time budget is spent.
start=$(date +%s); pass=0; bench_fail=0
while (( $(date +%s) - start < STRESS_MINUTES * 60 )); do
    vllm bench serve --backend vllm --model "$MODEL" --port 8000 \
        --dataset-name random --random-input-len 4096 --random-output-len 32 \
        --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONCURRENCY" \
        --seed 0 --ignore-eos > "$OUT/bench_$pass.log" 2>&1 || { bench_fail=1; echo "bench pass $pass failed"; break; }
    kill -0 "$VLLM" 2>/dev/null || { echo "vLLM died during pass $pass"; break; }
    pass=$((pass + 1))
done

engine_alive=0; kill -0 "$VLLM" 2>/dev/null && engine_alive=1
vllm_errors=$(grep -ciE "hipError|EngineDead|Traceback|invalid argument|Memory access fault|HSA_STATUS_ERROR" "$OUT/vllm.log")
server_errors=$(grep -ciE "ERROR|Traceback|hipError|Memory access fault" "$OUT/server.log")
stores=$(grep -c "Stored .* tokens" "$OUT/server.log"); retrieves=$(grep -c "Retrieved .* tokens" "$OUT/server.log")
echo "=== ROCm 10 MP stress summary ==="
echo "passes=$pass stores=$stores retrieves=$retrieves engine_alive=$engine_alive bench_fail=$bench_fail vllm_errors=$vllm_errors server_errors=$server_errors"
grep -iE "hipError|EngineDead|invalid argument|Memory access fault" "$OUT/vllm.log" | head -5
grep -iE "ERROR|hipError|Memory access fault" "$OUT/server.log" | head -5
kill "$VLLM" "$SERVER" 2>/dev/null; sleep 5; kill -9 "$VLLM" "$SERVER" 2>/dev/null
[[ "$engine_alive" == 1 && "$bench_fail" == 0 && "$vllm_errors" == 0 && "$server_errors" == 0 && "$pass" -ge 1 ]]
