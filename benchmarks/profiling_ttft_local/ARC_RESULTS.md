# Existing Local Policy Versus Stock ARC

Descriptive GPU observation on the frozen historical implementation, before
the probability-model changes. This is not current-PR-head performance or
evidence that the new formula model improves TTFT.

| Independent Restart Pair | ARC Mean TTFT ms | Existing Local Policy ms | Reduction |
|---|---:|---:|---:|
| 0: ARC then local | 62.3002 | 51.9320 | 16.64% |
| 1: local then ARC | 62.6147 | 51.7649 | 17.33% |
| 2: ARC then local | 60.5592 | 51.9490 | 14.22% |
| All evaluation requests | **61.8247** | **51.8820** | **16.08%** |

Hot-class mean: 50.3012 to 33.7010ms. Scan-class mean: 73.3482 to 70.0629ms.
Evaluation p95 (floor order statistic): 77.1259 to 71.8716ms. Each role has 144
evaluation requests; 360/360 requests completed, including warmup. All 180 paired
eight-token outputs agree. [Sanitized per-request data](arc_results.json).

Both roles use the same original three traces and offered 1.1s schedule, RTX5060Ti,
BF16, TP1/C1, APC off, 384MiB GPU KV, 0.5GiB L1, 4GiB filesystem L2. Policy
`a101de47`, stock ARC baseline `d0a86807`, producer `a20056de`. Costs remain fixed
illustrative parameters. The ARC implementation also exists in the newer dev
base. Source, commands, model/dependency/native fingerprints and actual selected
policies were independently checked in the retained packets.

Across all requests, L1-hit tokens ARC/local: 98048/156160; L2-hit tokens:
51456/0; completed L2-store chunks: 288/0. This compares combined retention and
write admission, not the value equation or individual signals in isolation.

Limitations: stable prewarmed hotspots, fixed-size equal-length documents,
mechanism-friendly capacity, only three paced C1 pairs, no production or
signal-ablation qualification. All six arms have CUDA shared-tensor teardown
warnings; block0 ARC additionally logs EngineDeadError during shutdown. Owner
waits were unforced but clean GPU-resource teardown is NOT qualified. Six known
AppleDouble metadata sidecars prevent literal common_source dictionary equality;
tracked Python parity is checked separately, not silently called byte-identical.
No specific ARC callback/eligibility defect was established by the bounded audit.
Prior LRU numbers are a different campaign, not a contemporaneous ARC/LRU ranking.

Raw evidence archive SHA256:
`d3809d1f9cf56be0f6429f4c861d693637280924cae4244ca424e58845dbb246`.
