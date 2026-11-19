# Exploratory Local-Policy TTFT Observation

This is historical, source-bound PoC evidence, not a benchmark of this rebased
PR head or a general LMCache speedup claim.

| Restart Pair | Unmodified LRU (ms) | Local Policy (ms) | Reduction |
|---|---:|---:|---:|
| 0: LRU then full | 60.1554 | 51.8740 | 13.77% |
| 1: full then LRU | 65.3301 | 51.8821 | 20.58% |
| 2: LRU then full | 64.2716 | 52.3367 | 18.57% |
| Mean over all evaluation requests | 63.2524 | 52.0310 | **17.74%** |

360/360 requests completed. Each arm/block has12 warmup and48 evaluation
requests; all144 evaluation requests per role enter the mean. All180 paired
responses, including warmup, have identical generated token IDs. Evaluation p95:
77.7864ms LRU /72.1276ms full. Maximum evaluation dispatch lag is below3.5ms.
[Sanitized per-request timings and output digests](results.json) allow independent
recalculation; private paths, hosts, environment inventories and prompts are not
published. This export is not a complete clean-checkout reproducer.

## Why This Workload Improves

Two reusable1536-token prefixes alternate with unique scans. LRU retains recent
scans and loses parts of valuable prefixes. The local policy protects observed
hot prefixes and avoids low-value L2 writes. Aggregate L1-hit tokens increase
from91392 to156672; L2-hit tokens fall from47360 to0. Baseline completed342 L2
store chunks; treatment completed0. These counters support the admission/
retention mechanism, not an independent causal decomposition of the17.74%.

Hardware/settings: RTX5060Ti16GB, Qwen3 architecture (28 layers, hidden1024),
BF16, TP1/C1, APC off, eager TRITON_ATTN,384MiB GPU KV,0.5GiB host L1,4GiB
filesystem L2 in both roles. Both use the same1.1s offered request schedule,
allowing the native1s eviction loop to run. No artificial storage delay.
Text is synthetic and prompts instruct a fixed sentence: output equality is a
paired generation check, not a context-dependent task-quality or tensor oracle.

## Provenance And Limits

Measured unmodified base: `d0a8680792e319e00947ceb55f06a0097736ee67`.
Measured policy: `a101de47`; producer: `a20056de` in the retained experiment.
The seven policy/integration-file hashes matched the experiment source. Model,
native, runtime, engine commands and payload capacities matched across roles.
Independent raw-packet review reproduced the arithmetic, checked180 paired
outputs and rejected six private invalid-packet mutations.
Original raw evidence archive SHA256:
`27b2eac17886effc801726299085f1d1141584dc5a31cd47242f1006ae6df512`.

The costs were fixed **illustrative** parameters: for prefix length i chunks,
R=12i,F1=3i,F2=7i,W=i milliseconds (i=1..8), not accepted measured calibration.
Earlier calibration attempts had reproducible recompute/restore output drift;
these failures remain unresolved and were not converted into valid cost data.
The earlier fast-paced development run observed only about1.56% arithmetic
improvement and failed its final policy-name health check; it is not included
in this separately frozen three-pair experiment.

No eight-pair qualification, signal ablations, representative/all-fit/low-reuse
controls or total-host-memory-envelope validation was performed. Costs and
workload are mechanism-friendly; no Dell/customer or patent-filing claim follows.
Allocation warnings and CUDA IPC teardown warnings remain; one baseline logs
an output-handler error after shutdown initiation. All owners exited unforced,
but complete resource cleanup is not qualified. The disposable runner also has
a conditional timeout-cleanup gap and is deliberately not published as a reusable
benchmark tool. New-dev GPU correctness/performance and public CI remain pending.
