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
The `no-cost` ablation retains its historical count-domain threshold and requires
`value_model="expected_reuses"`; probability mode rejects that combination.

By default, L2 admission has a zero opportunity-cost floor and native L2 eviction
is unchanged. Opt-in `l2_capacity_policy="value"` prices each incoming block
against the lowest-value eligible resident leaf when capacity is full; resident
write costs are sunk, while incoming blocks still pay write cost. The native
owner supplies completed/pending inventory and executes deletion; the controller
refreshes and re-plans before submitting. This requires initially empty,
exclusive local FS with finite capacity, uniform blocks and no padding. Unknown
or stale value-mode data fails closed, rather than writing without a budget.
Predictions are heuristics, not
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

To isolate replacement from admission, select `--eviction-policy ARC
--profiling-store-config costs.json`. The independent admission observer receives
the same demand and L1 lifecycle notifications, without replacing native ARC.
Do not combine independent admission config with `ProfiledValue` L1 selection.

The experimental namespace is empty cache salt, TP1 and object group0. Unknown
identity/layout, stale costs or incomplete resident knowledge fall back to native
behavior in the default capacity mode; value-capacity writes fail closed instead.
Layout mismatch latches fallback. Runtime L2 mutations return HTTP409;
restart with a newly calibrated snapshot instead. Subscription is deferred until
StorageManager construction commits.

## Evidence

The latest bounded model experiment remains **FAIL**: six of 756 responses
disagree with the no-external-cache oracle, including warmup, and two jobs log
an output-handler EngineDeadError during shutdown. Native partial-prefix APC
also reproduces both alternate output sequences without LMCache; that does not
prove LMCache payload integrity or identify a common cause. The new probability
and capacity models have no qualified TTFT improvement claim. See the
[enhanced validation record](../../../../../benchmarks/profiling_ttft_local/ENHANCED_RESULTS.md)
for source versions, scope and remaining checks.

See [the bounded exploratory result](../../../../../benchmarks/profiling_ttft_local/RESULTS.md).
Its illustrative costs are **not** an accepted measured calibration. There are
no per-signal attribution, production-workload or current-dev GPU performance
claims. The new-dev port preserves the upstream L1 factory/write-policy APIs;
the old private constructor injection is not restored.

See [the formula follow-up contract](value_formula_followup.md) for the ARC-first
execution order, capacity-owner contract and acceptance tests. A fixed write
charge is not a capacity-value watermark.

## Relationship To External Policies

[PR #5786](https://github.com/LMCache/LMCache/pull/5786) provides explicit
`module:Class` loading for external StorePolicy and PrefetchPolicy classes.
This bundled experimental policy uses the existing interfaces directly and has
no runtime or Git ancestry dependency on that unmerged PR. Do not merge a second
external loader here. A future external-package version should reuse #5786;
its loader alone does not bind demand/L1 observation, share an instance across
admission and L1 eviction, or execute owner-safe capacity replacement.

The current implementation remains a local algorithm, not a general plugin
framework. Its formula, calibration and history can move to an external package.
Any required lifecycle/L1 extension should be proposed separately and kept
algorithm-neutral. Hard L2 capacity can alternatively belong to an external
adapter that owns inventory, reservations and completion. Neither externalization
nor general multi-tier/shared-storage support is delivered by this PR.
