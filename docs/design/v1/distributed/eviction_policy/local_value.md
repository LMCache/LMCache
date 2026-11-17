# Experimental Local Profiling Policy

`ProfiledValue` is opt-in. It shares one bounded demand history between native
L1 eviction and L2 store admission. Stock LRU/ARC remain the default choices.
Request lookup events, including misses, update decayed access frequency and
distinct-key reuse distance. Storage managers retain allocation, locking,
completion and deletion ownership.

## Decisions

For prefix increment c, predicted reuses N, recompute cost R, disk fetch F2,
host fetch F1 and disk write W:

```
first disk copy: N * (R - F2) - W
DRAM with known disk backing: N * (F2 - F1)
DRAM without known disk backing: N * (R - F1)
```

Costs are first differences of cumulative prefix tables, in milliseconds.
Frequency is decayed; distinct-key distance penalizes long reuse intervals.
Only positive-value known disk writes are selected. Required ancestors must be
selected or known backed. Backing knowledge comes from completed native L2
callbacks, never proposed writes. Victims still pass native eligibility checks.

Schema2 defaults retain the historical `value_model="expected_reuses"` and
`recompute_fraction=0`. Opt in to `value_model="reuse_probability"` for a bounded
`P = 1-exp(-lambda*H/penalty)` estimate of at least one future reuse. Lambda uses
decayed repeat observations divided by their exponentially weighted time
exposure; the initial observation is excluded. Distance penalizes the intensity.
This is an uncalibrated Poisson modeling assumption, not a measured probability
or a count of all repeated hits. In probability mode, substitute P for N above.

Finite `recompute_fraction` in [0,1] discounts R to `(1-r)*R` for first-copy and
unbacked-DRAM value. Disk-backed replica value remains `P*(F2-F1)`; the common
recomputation work cancels. This parameter supplies a cost assumption; it does
not enable partial recomputation or content blending in the inference engine.
Costs, probability model, horizon and r must be bound to each new experiment;
historical measurements did not execute this probability model.

The prototype does not price L2 capacity opportunity cost: its admission floor
is zero and native L2 eviction is unchanged. Predictions are heuristics, not
calibrated reuse probabilities. It operates **after D2H** and cannot eliminate
GPU-to-host copies. No coordinator, fleet placement, transport or load-versus-
recompute routing is introduced.

## Configuration And Scope

Select `--eviction-policy ProfiledValue --profiling-config costs.json` on the MP
server, with exactly one L1, one fixed L2 and default store policy. Schema2 JSON
requires `model_name` (local model directory), `chunk_size`, `model_revision`
(SHA256 of the sorted relative-file-to-SHA256 inventory), `dtypes`, `shapes`, and
equal-length monotone nonnegative arrays `prefix_recompute_ms`,
`prefix_fetch_l1_ms`, `prefix_fetch_l2_ms`, `prefix_write_l2_ms`.
Model bytes must match. `LocalValueConfig` documents optional lifetime/history
settings and modes. Configured arrays alone are not measurement provenance.

The experimental namespace is empty cache salt, TP1 and object group0. Unknown
identity/layout, stale costs or incomplete resident knowledge fall back to native
behavior. Layout mismatch latches fallback. Runtime L2 mutations return HTTP409;
restart with a newly calibrated snapshot instead. Subscription is deferred until
StorageManager construction commits.

## Evidence

See [the bounded exploratory result](../../../../../benchmarks/profiling_ttft_local/RESULTS.md).
Its illustrative costs are **not** an accepted measured calibration. There are
no per-signal attribution, production-workload or current-dev GPU performance
claims. The new-dev port preserves the upstream L1 factory/write-policy APIs;
the old private constructor injection is not restored.

See [the formula follow-up contract](value_formula_followup.md) for the ARC-first
execution order, acceptance tests and unresolved owner-level opportunity-cost
interface. A fixed write charge is not a capacity-value watermark.
