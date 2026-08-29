#!/bin/bash
# End-to-end A/B for the raw_block io_uring_cmd load path: bring up a real
# LMCache MP server (raw_block L2 on an NVMe passthrough char device) + vLLM,
# drive TTFT at low and higher concurrency, and emit validity guards that prove
# the loads were actually served from the device (not the L1 DRAM cache and not
# GPU recompute). See README.md for the four confounds these guards defend.
#
# Usage:  ab_arm.sh <arm-label>
#   The script copies "$CORE_VARIANT_DIR/core_<arm-label>.py" over the live
#   raw_block core.py, so prepare two variants to compare, e.g.
#     core_stock.py    -- baseline (per-chunk read_uring, QD~1)
#     core_batched.py  -- batched_read()+wait_iouring() per object
#   Upstream you would instead compare two builds/branches; the in-place toggle
#   is just a convenience for a single checkout.
#
# All host-specific values below are overridable via the environment.
set -u
ARM="$1"

# ---- environment-specific config (override via env) -------------------------
: "${DEVICE:=/dev/ng1n1}"                 # NVMe passthrough char device (io_uring_cmd)
: "${NVME_CTRL:=/dev/nvme1}"              # its controller, for smart-log counters
: "${MODEL:=meta-llama/Llama-3.1-8B-Instruct}"
: "${SLOT_BYTES:=33685504}"              # per-object slot (KV chunk + 128KiB header)
: "${L1_SIZE_GB:=24}"                    # DRAM pool: >= concurrent staging, << working set
: "${MQ_TIMEOUT:=10}"                    # connector load timeout (s); 10=realistic, 90=isolate storage
: "${MAX_WORKERS:=4}"
: "${MP_PORT:=6555}"; : "${HTTP_PORT:=8080}"; : "${VLLM_PORT:=8000}"
: "${CORE_PATH:=$HOME/kvio/LMCache/lmcache/v1/storage_backend/raw_block/core.py}"
: "${CORE_VARIANT_DIR:=$HOME/kvio}"      # holds core_<arm>.py variants
: "${BIN:=}"                             # optional venv bin dir prefix, e.g. ~/venv/bin
: "${OUTDIR:=$HOME/kvio-bench/ab}"
: "${DRIVER:=$(dirname "$0")/ab_drive.py}"
lmcache() { ${BIN:+$BIN/}lmcache "$@"; }
vllm()    { ${BIN:+$BIN/}vllm "$@"; }
py()      { ${BIN:+$BIN/}python3 "$@"; }
# -----------------------------------------------------------------------------

D="$OUTDIR"; mkdir -p "$D"; RES="$D/result_$ARM.txt"
export LMCACHE_TRACK_USAGE=false
: > "$RES"; echo "==== ARM=$ARM $(date +%T)  mq_timeout=$MQ_TIMEOUT ====" >> "$RES"

# kill any prior stack (use exact patterns, never a bare 'vllm' that self-matches)
pkill -f "bin/vllm" 2>/dev/null; pkill -f "bin/lmcache server" 2>/dev/null; sleep 6
pkill -9 -f "bin/vllm" 2>/dev/null; sleep 2
sudo chown "$USER" "$DEVICE" 2>/dev/null || true

# select the arm's core.py variant
cp "$CORE_VARIANT_DIR/core_$ARM.py" "$CORE_PATH"
echo "core.py <- core_$ARM.py ($(grep -c 'batched_read(chunk_offsets' "$CORE_PATH") batched-markers)" >> "$RES"

CUDA_VISIBLE_DEVICES=0 \
nohup lmcache server --l1-size-gb "$L1_SIZE_GB" --l1-align-bytes 131072 \
  --eviction-policy LRU --l2-store-policy skip_l1 --l2-prefetch-policy default \
  --l2-adapter "{\"type\":\"raw_block\",\"device_path\":\"$DEVICE\",\"slot_bytes\":$SLOT_BYTES,\"block_align\":131072,\"header_bytes\":131072,\"io_engine\":\"io_uring\",\"use_uring_cmd\":true,\"load_checkpoint_on_init\":false}" \
  --max-workers "$MAX_WORKERS" --port "$MP_PORT" --http-port "$HTTP_PORT" > "$D/mp_$ARM.log" 2>&1 &
sleep 12
ss -ltn | grep -q "$MP_PORT" || { echo "MP FAILED" >> "$RES"; tail -5 "$D/mp_$ARM.log" >> "$RES"; exit 1; }

env -u VLLM_PORT CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
nohup vllm serve "$MODEL" \
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.mq_timeout\":$MQ_TIMEOUT}}" \
  --attention-backend FLASH_ATTN --no-enable-prefix-caching \
  --port "$VLLM_PORT" --no-async-scheduling > "$D/vllm_$ARM.log" 2>&1 &

UP=0
for i in $(seq 1 96); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:$VLLM_PORT/health 2>/dev/null)" = 200 ] && { UP=1; break; }
  sleep 5
done
[ "$UP" = 1 ] || { echo "VLLM FAILED" >> "$RES"; tail -8 "$D/vllm_$ARM.log" >> "$RES"; exit 1; }
echo "stack up $(date +%T)" >> "$RES"

# The authoritative device-read counter is the NVMe controller's Data Units Read
# (512KB units). It COUNTS io_uring_cmd passthrough, which block-layer iostat /
# /proc/diskstats do NOT -- so it is how we prove loads hit the device.
MPLINE=$(wc -l < "$D/mp_$ARM.log"); VLINE=$(wc -l < "$D/vllm_$ARM.log")
duread() { sudo nvme smart-log "$NVME_CTRL" -o json 2>/dev/null \
    | py -c "import sys,json;print(json.load(sys.stdin)['data_units_read'])"; }
: > "$D/devgb_$ARM.txt"
drive_one() { # $1=concurrency $2=duration_s
  u0=$(duread)
  py "$DRIVER" --arm "$ARM" --concurrency "$1" --duration "$2" 2>&1 \
    | grep -E "RESULT|warmed|NO_COMPLETED" >> "$RES"
  u1=$(duread)
  echo "  C=$1 device GB read = $(py -c "print(f'{($u1-$u0)*0.000512:.2f}')")" >> "$D/devgb_$ARM.txt"
}
drive_one 1 45
drive_one 4 30

# ---- validity guards (README.md explains each) ------------------------------
echo "--- external prefix cache hit rate (LOW ~0 => GPU recompute, not cache) ---" >> "$RES"
tail -n +"$VLINE" "$D/vllm_$ARM.log" | grep -oE "External prefix cache hit rate: [0-9.]+%" | sort -t: -k2 -n | tail -3 >> "$RES"
echo "--- alloc failures (>0 => staging pool starved => silent recompute) ---" >> "$RES"
echo "  $(tail -n +"$MPLINE" "$D/mp_$ARM.log" | grep -c 'Failed to batched allocate') alloc failures" >> "$RES"
echo "--- device GB read per pass (NVMe ctrl counter; counts passthrough) ---" >> "$RES"
cat "$D/devgb_$ARM.txt" >> "$RES"
echo "--- tier split (want 0 L1 => no DRAM-cache dilution) ---" >> "$RES"
tail -n +"$MPLINE" "$D/mp_$ARM.log" | grep -oE "retained keys \([0-9]+ L1, [0-9]+ L2\)" | sort | uniq -c | tail -6 >> "$RES"

pkill -f "bin/vllm" 2>/dev/null; pkill -f "bin/lmcache server" 2>/dev/null; sleep 5
pkill -9 -f "bin/vllm" 2>/dev/null
echo "ARM_DONE $ARM $(date +%T)" >> "$RES"
