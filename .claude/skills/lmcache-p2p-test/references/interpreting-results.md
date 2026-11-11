# Interpreting P2P test results

**What a pass looks like.** 0 errors in every phase; peer-phase hit ratio at the chunk-aligned
maximum (`floor(prompt_tokens / 256) x 256 / prompt_tokens`, about 0.98 for 7k tokens);
identical outputs in basic_test; peer TTFT well below the warm (cold) TTFT at the same
concurrency (2-5x on H200 for 7k-23k-token prompts); `l2_load_completed{l2_name="p2p"}`
growing on the reading node and `active_p2p_lookup_jobs` briefly non-zero on the serving node.

**Hit ratio below the maximum, 0 errors.** Read
`lmcache_mp_l2_prefetch_failure_chunks_total{reason}`:
- `l1_oom`: the reader could not reserve temporary L1 for the load. Each in-flight P2P load
  needs prompt-sized L1 (27 chunks x 37.7 MB = 1 GB per 7k-token Qwen3-8B request). With 32
  in flight on a 20 GB L1 the hit ratio fell to 0.84 and p99 TTFT to 2.4 s; 80 GB fixed both.
- `l1_contended`: two requests for the same not-yet-local document on one node at the same
  time; the second recomputes. A few % in the mixed phases is normal.
- Misses only in the first seconds of the peer phase: the `--gap` after warm was too short for
  the home node to finish storing.

**Peer TTFT not better than cold.** Check `cached_tokens` in the CSV: all zero means no P2P
(go to the diagnose skill); non-zero but slow means the transfer is the bottleneck (single
rail? see `lmcache-p2p-launch/references/ucx-multirail.md`) or the reader is saturated
(`vllm_wait` high, L1-to-GPU copies). At 16 in flight with 15k-token prompts the bench shows
P2P within about 90 ms of a local L1 hit: the fabric is no longer the limiting factor there.

**Per-load GB/s vs real bandwidth.** The histogram mean (`load_GB/s` column) is bytes over
submit-to-complete latency per request. It rose from 14-40 to about 120 GB/s when going from
one to eight RoCE rails on H200 nodes, but it includes queueing and lookup time, so it
understates the wire rate under heavy overlap and is not comparable across workloads. Use
the NIC counter diff for absolute numbers.

**Bench numbers drift between runs.** `lmcache bench` seeds its documents, so after the first
run both nodes may already hold them (A from populate, B from its own stores after pass 2),
and the 80 GB L1 evicts older documents as the stress ladder writes through it. Compare
configurations with runs taken at the same point in that history, or clear L1 first.

**vLLM log traceback right after a restart.** The previous vLLM API server can write its
shutdown traceback (`Timed out waiting for engine core`) into the freshly truncated log;
check the process start time before treating it as a failure.
