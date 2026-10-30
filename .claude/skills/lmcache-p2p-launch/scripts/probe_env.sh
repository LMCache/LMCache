#!/bin/bash
# Probe one host for everything a P2P deployment needs and print a report.
# usage: probe_env.sh [--python PYTHON] [--model HF_ID] [--hf-home DIR] [--ports "6700 7700 8700 8300 12500"]
# Exit code is 0 even when something is missing; read the WARN lines.
PY=python; MODEL=""; HFH="${HF_HOME:-$HOME/.cache/huggingface}"; PORTS="6700 7700 8700 8300 12500"
while [ $# -gt 0 ]; do case "$1" in
  --python) PY=$2; shift 2;; --model) MODEL=$2; shift 2;; --hf-home) HFH=$2; shift 2;; --ports) PORTS=$2; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
warn() { echo "WARN: $*"; }
echo "== host: $(hostname)  ip: $(hostname -I 2>/dev/null | awk '{print $1}')  numa nodes: $(lscpu 2>/dev/null | awk -F: '/NUMA node\(s\)/{gsub(/ /,"",$2); print $2}')"
echo "== GPUs"
if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
else warn "nvidia-smi not found"; fi
echo "== RDMA devices (device, link layer, state, NUMA node)"
if command -v ibv_devices >/dev/null; then
  for d in /sys/class/infiniband/*; do [ -e "$d" ] || continue
    printf "  %-14s %-9s %-7s numa=%s\n" "$(basename "$d")" "$(cat "$d"/ports/1/link_layer 2>/dev/null)" "$(cat "$d"/ports/1/state 2>/dev/null | awk '{print $2}')" "$(cat "$d"/device/numa_node 2>/dev/null)"
  done
  n=$(ls -d /sys/class/infiniband/* 2>/dev/null | wc -l); [ "$n" -gt 0 ] || warn "no RDMA devices: P2P will fall back to TCP (functional only)"
  lag=$(ls -d /sys/class/infiniband/*bond* 2>/dev/null | wc -l); [ "$lag" -gt 0 ] && echo "  note: $lag LAG bond devices -> each counts as 2 UCX paths (see references/ucx-multirail.md)"
else warn "ibv_devices not found (rdma-core missing?)"; fi
echo "== tools"
for t in numactl ib_write_bw rsync docker uv; do command -v $t >/dev/null && echo "  $t: $(command -v $t)" || echo "  $t: MISSING"; done
echo "== python stack ($PY)"
if command -v "$PY" >/dev/null; then "$PY" -P - <<'PY' 2>/dev/null
import importlib
def v(m):
    try:
        mod = importlib.import_module(m); return getattr(mod, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        return f"MISSING ({type(e).__name__})"
import sys; print("  python", sys.version.split()[0])
for m in ("torch", "vllm", "lmcache", "nixl", "prometheus_client", "aiohttp"): print(f"  {m}: {v(m)}")
try:
    import torch; print("  torch.cuda", torch.version.cuda, "available", torch.cuda.is_available())
except Exception: pass  # noqa: BLE001
try:
    import lmcache.cuda_ops; print("  lmcache.cuda_ops: ok")
except Exception as e: print(f"  lmcache.cuda_ops: MISSING ({e})")  # noqa: BLE001
PY
  CLI=$(command -v lmcache || ls "$(dirname "$(command -v "$PY")")/lmcache" 2>/dev/null)
  if [ -n "$CLI" ]; then echo "  lmcache CLI: $CLI; p2p flags: $("$CLI" server --help 2>/dev/null | grep -c -- '--p2p')"; else warn "lmcache CLI not found (not on PATH nor next to $PY)"; fi
else warn "$PY not found"; fi
echo "== ports ($PORTS)"
for p in $PORTS; do ss -ltn 2>/dev/null | grep -q ":$p " && warn "port $p is in use" || echo "  $p free"; done
if [ -n "$MODEL" ]; then
  echo "== model $MODEL"
  hub="$HFH/hub/models--${MODEL//\//--}"
  if [ -d "$hub/snapshots" ]; then echo "  cached at $hub ($(du -sh "$hub" 2>/dev/null | cut -f1))"; else warn "model not in HF cache ($hub); download it or sync it from another node"; fi
fi
