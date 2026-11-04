---
name: lmcache-p2p-test
description: Test a running LMCache P2P deployment: cross-node correctness (basic_test), stress ladder (concurrency, long documents, sustained mixed load), the standard lmcache bench long-doc-qa across nodes, and live Prometheus monitoring with a results summary. Use when asked to verify, stress, benchmark or monitor P2P KV sharing or to produce a P2P test report.
allowed-tools: Bash, Read, Grep, Glob, Write
argument-hint: "[basic | stress | bench | monitor | all] [--concurrency N] [--tag NAME]"
---

# LMCache P2P: correctness, stress, bench, monitoring

Needs a deployment brought up by `lmcache-p2p-launch` (its `env.sh` is read by every script
here through `$P2P_ENV` / `$P2P_HOME`). Node A is the instance in `env.sh`, node B its peer.
Run the scripts with the deployment's Python (vLLM's venv: it has `aiohttp` and `lmcache`).
Results land in `$P2P_HOME/results/`. If any run shows errors, peer count 0 or a hit ratio
near zero, switch to `lmcache-p2p-diagnose`.

Two facts shape every client here: vLLM returns `kv_transfer_params.cached_token_stats`
**only on non-streaming responses**, so TTFT is measured as the latency of a `max_tokens=1`
request; and P2P-loaded keys are **not retained** in the reader's L1 by default, so every
repeat of a peer's prompt is another P2P transfer (good for stress, surprising in a demo).

Verified with LMCache dev `45a0bb420`, vLLM 0.32.0, nixl 1.5.2, two H200 nodes, Qwen/Qwen3-8B;
reference numbers below come from that setup with an 80 GB L1 and 8 RoCE rails.

## 1. Basic correctness (first, always)

`scripts/basic_test.py` : fresh 7k-token prompt cold on A, same prompt on B, outputs compared,
then the reverse. PASS needs `num_lmcache_cached_tokens >= 0.8 x prompt_tokens` on the peer and
identical 16-token outputs. Reference: cold 0.20 s -> peer 0.08 s TTFT, 6912/7032 tokens.
Zero cached tokens with a 2x TTFT drop means the client is streaming (stats missing), not a
P2P failure: check `/metrics` `lmcache_mp_l2_load_completed_requests_total{l2_name="p2p"}`.

## 2. Stress ladder (`scripts/stress_p2p.py`)

Phases: `warm` (cold on home node) -> `peer` (every query forced over P2P) -> `mixedN`.
Pass = 0 errors and hit ratio at the chunk-aligned maximum (about 0.98 for 7k-token prompts,
0.99 for 23k). Run in this order and stop at the first failing rung:

| rung | args | checks | reference (peer p50 TTFT, prefill tok/s) |
|---|---|---|---|
| moderate | `--docs 16 --doc-words 6000 --queries-per-doc 3 --concurrency 8` | plumbing under light overlap | 0.14 s, 310k |
| high concurrency | `--docs 24 --doc-words 6000 --queries-per-doc 4 --concurrency 32 --mixed-rounds 2` | L1 sizing (`l1_oom` fallbacks if too small) | 0.38 s, 544k |
| long documents | `--docs 8 --doc-words 20000 --queries-per-doc 3 --concurrency 8` | 3.4 GB transfers per request | 0.38 s, 360k |
| sustained | `--docs 32 --doc-words 6000 --queries-per-doc 6 --concurrency 16 --mixed-rounds 6` | stability over 1.4k requests | 0.21 s, 480k |

Give each run `--tag` so its JSON/CSV and Prometheus export are identifiable. A hit ratio
below the maximum with 0 errors is a graceful fallback; read the reason in
`lmcache_mp_l2_prefetch_failure_chunks_total{reason=...}` (`l1_oom`: grow `L1_GB`;
`l1_contended`: the same document fetched concurrently on one node, expected at a few %).

## 3. Standard benchmark (`scripts/bench_ldqa.sh`)

`lmcache bench engine` long-doc-qa: populate A, then query B (all P2P), optionally
`--with-local-baseline` to run the same workload on A (local L1 hits) for comparison.
`tokens_per_gb.py` derives `--tokens-per-gb-kvcache` from the model config (the server reports
models by snapshot path, which breaks `--lmcache-url` auto-detection). Reference at 40 GB /
15k-token docs / 16 in flight: B p50 0.73 s vs A-local 0.64 s; failed 0. The 80 GB L1 holds
about 40 GB of documents, so run the bench before the stress ladder or clear L1
(`lmcache kvcache clear --url $HTTP_URL`) if you need a truly cold populate pass.

## 4. Monitoring

`scripts/start_prometheus.sh` (docker, host network, UI on :9710) scrapes both LMCache
servers, the coordinator and both vLLMs every 2 s. `scripts/monitor.py` prints a per-node
table every 5 s and appends `results/monitor.csv`; run it in the background during the ladder.
`scripts/export_prom.py --from-json results/stress_<tag>.json results/prom_<tag>.csv` saves the
series for a run. Metric meanings and PromQL: `references/metrics.md`. The per-load
throughput histogram is bytes over submit-to-complete latency, not link bandwidth: use it
to compare configurations, and NIC counters (diagnose skill) for absolute bandwidth.

## 5. Report

`scripts/summarize.py --out results/REPORT.md` tabulates every stress JSON, bench directory and
Prometheus export. Add: the setup (versions, L1 size, rails), what each rung proved, the
failure reasons seen, and anything left unverified. How to read the numbers:
`references/interpreting-results.md`.

## Eval prompts

- "Check that P2P actually works between the two nodes" -> basic_test, report cached tokens
  and the `l2_name="p2p"` counter on the peer.
- "Stress P2P at concurrency 32" -> high-concurrency rung with monitor running; if hit < 0.9,
  quote the failure reason and recommend an `L1_GB`.
- "Compare single-rail vs 8-rail" -> same tags with `_rails1`/`_rails8`, `summarize.py`.
