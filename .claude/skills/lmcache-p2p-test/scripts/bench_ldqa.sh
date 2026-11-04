#!/bin/bash
# Standard `lmcache bench engine` long-doc-qa across the two nodes.
# Pass 1 populates node A's L1 (one query per document); pass 2 runs the same seeded document
# set against node B, so every prefill there is a P2P read from A. Optional pass 3 runs the
# pass-2 workload against A again (local L1 hits) as the baseline to compare B against.
# usage: bench_ldqa.sh [--volume-gb 40] [--doc-len 15000] [--qpd 4] [--inflight 16] [--tag NAME] [--with-local-baseline]
#        [--tokens-per-gb N]   (default: computed from the model config by tokens_per_gb.py)
source "$(dirname "$0")/../../lmcache-p2p-launch/scripts/common.sh"
D="$(cd "$(dirname "$0")" && pwd)"
VOL=40; DOCLEN=15000; QPD=4; INF=16; TAG=ldqa; TPG=""; BASELINE=0
while [ $# -gt 0 ]; do case "$1" in
  --volume-gb) VOL=$2; shift 2;; --doc-len) DOCLEN=$2; shift 2;; --qpd) QPD=$2; shift 2;; --inflight) INF=$2; shift 2;;
  --tag) TAG=$2; shift 2;; --tokens-per-gb) TPG=$2; shift 2;; --with-local-baseline) BASELINE=1; shift;;
  *) echo "unknown arg $1" >&2; exit 2;; esac; done
# --lmcache-url auto-detection fails when the server reports the model by snapshot path, so pass the value.
TPG=${TPG:-$(python "$D/tokens_per_gb.py" "$MODEL")} || exit 1
OUT="$RESULTS_DIR/bench_$TAG"; mkdir -p "$OUT"; echo "tokens_per_gb=$TPG out=$OUT"
COMMON=(--workload long-doc-qa --model "$MODEL" --tokens-per-gb-kvcache "$TPG" --kv-cache-volume "$VOL"
        --ldqa-document-length "$DOCLEN" --ldqa-shuffle-policy random --ldqa-max-output-length 1 --seed 0
        --no-interactive --json --quiet)
export LMCACHE_DISABLE_BANNER=1
date +%s > "$OUT/start.ts"
echo "== pass 1: populate A ($VLLM_URL)"
lmcache bench engine --engine-url "$VLLM_URL" "${COMMON[@]}" --ldqa-query-per-document 1 --ldqa-num-inflight-requests "$INF" --output-dir "$OUT/populate_A" 2>&1 | grep -E "Documents|Total requests|ERROR|Traceback"
echo "== pass 2: query B ($PEER_VLLM_URL): $QPD queries/doc, $INF in flight -> P2P reads from A"
lmcache bench engine --engine-url "$PEER_VLLM_URL" "${COMMON[@]}" --ldqa-query-per-document "$QPD" --ldqa-num-inflight-requests "$INF" --output-dir "$OUT/query_B" 2>&1 | grep -E "Total requests|ERROR|Traceback"
if [ "$BASELINE" = 1 ]; then
  echo "== pass 3: same workload on A (local L1 hits) as baseline"
  lmcache bench engine --engine-url "$VLLM_URL" "${COMMON[@]}" --ldqa-query-per-document "$QPD" --ldqa-num-inflight-requests "$INF" --output-dir "$OUT/query_A_local" 2>&1 | grep -E "Total requests|ERROR|Traceback"
fi
date +%s > "$OUT/end.ts"
for d in populate_A query_B query_A_local; do [ -f "$OUT/$d/bench_summary.json" ] || continue
  python - "$OUT/$d/bench_summary.json" "$d" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))["results"]
print(f"  {sys.argv[2]:14} requests={r['total_requests']} failed={r['failed_requests']} ttft mean/p50/p90/p99 ms="
      f"{r['mean_ttft_ms']:.0f}/{r['p50_ttft_ms']:.0f}/{r['p90_ttft_ms']:.0f}/{r['p99_ttft_ms']:.0f} input_tok/s={r['input_throughput']:.0f}")
PY
done
