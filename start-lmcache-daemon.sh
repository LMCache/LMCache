#!/bin/bash
# LMCache MP daemon startup script for GLM-5.2 DSA testing on node 43.
# Runs inside the sgl-lmcache container (as root) so it can access /ddn.
set -x

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_OPT_USE_TOPK_V2=false

python -m lmcache.v1.multiprocess.server \
  --host 127.0.0.1 --port 5556 --chunk-size 256 \
  --max-workers 4 --max-cpu-workers 16 \
  --l1-size-gb 10 --eviction-policy LRU \
  --l2-adapter '{"type":"fs","base_path":"/ddn/glm5.2.lmcache-dsa-test"}' \
  --disable-observability \
  2>&1 | tee /tmp/lmcache-daemon.log
