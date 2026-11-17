# Local Value Formula Follow-Up

User request (2026-11-17): first compare the unchanged measured policy against
stock ARC; then implement the pooled-KV value formula where the current local
interfaces support it, explicitly reporting remaining difficulties.

## Contract And Scope

The architectural source is `19_pooled_kvcache_architecture_zh.md` section4.2:
`P_reuse * ((1-r) * recompute - fetch) - storage opportunity cost`.
This follow-up is local policy, not fleet placement or a new transport.

1. Freeze the ARC experiment before modifying product behavior: same historical
   policy a101de47, stock d0a86807 with ARC, producer a20056de, same three traces,
   illustrative costs, capacities, C1/APC-off and offered schedule. Keep complete
   failures and every nonwarmup request; no outcome-dependent tuning or stopping.
2. Preserve schema2/default behavior for existing cost snapshots. Add opt-in
   `value_model="reuse_probability"`; the existing default remains
   `"expected_reuses"`. This prevents old results from silently describing a new
   policy. New formula results must name their source/model separately.
3. The probability-mode weight is `P = 1-exp(-lambda*H/penalty)`. With decay
   beta=ln(2)/half_life, lambda is the decayed repeat count divided by observed
   exponentially weighted time exposure `(1-exp(-beta*age))/beta`. Subtract the
   initial observation's decayed contribution, because no earlier exposure is
   known. A first observation alone gives zero; coincident repeated timestamps
   use the positive-rate limit P=1. Distinct-key distance supplies the existing
   penalty. This is bounded in [0,1], assumes a Poisson count model, and is NOT
   empirically calibrated. No future request labels are inputs. It estimates at
   least one future reuse, not total repeated-hit savings over H.
   The historical no-cost ablation remains count-domain only; reject it with
   probability mode rather than silently rejecting every known L2 write.
4. Add finite `recompute_fraction` in [0,1], default0, to discount recompute
   savings. This is a cost input supplied by the operator, not an instruction
   that makes the engine recompute that fraction. No content-blending engine is
   implemented. When a disk copy exists, its DRAM replica value is still the
   disk-versus-host fetch difference: the common recompute component cancels.
5. Exercise the weight and r through real store selection and eviction ranking;
   preserve prefix dependency, unsupported/stale fallback, locks, callbacks and
   native ownership. Do not add a detached unused formula helper.

## Alternatives And Limits

Replacing every snapshot's estimator would silently invalidate compatibility and
historical evidence. Adding arbitrary fixed storage penalties would not implement
the design's resident-value watermark. Both are rejected.

The first formula revision only received configuration-only L2 descriptors;
separate usage snapshots/listings did not provide a coherent reserve/replace
decision shared with native L2 eviction. Bounded completion history was not an
authoritative inventory, so that revision did not enforce dynamic opportunity cost.
That revision retained the zero opportunity-cost floor, explicit one-time write
charge and native L2 eviction. The following opt-in extension supplies a bounded
owner contract instead. L1 ordering is still not pre-allocation L1 admission.

## Dynamic Opportunity Cost Extension

Approved follow-up: `l2_capacity_policy="value"` (default remains `"native"`).
The native adapter supplies an immutable snapshot of its existing completed-key
size ledger, unique pending-write bytes and unavailable keys. The policy does
not maintain a second authoritative residency index. Only cold, initially empty,
exclusive local FS with finite capacity and uniform unpadded blocks qualifies;
preexisting files, external writers, shared FS, serde and variable-sized objects
are outside this contract. Capacity is managed KV payload, not filesystem space
including staging files or metadata.

For a completed resident block, write cost is sunk:
`resident_value = weight * ((1-r)*recompute - fetch_L2)`.
For an incoming block, `incoming_value = resident_value - write_L2`.
When capacity is free, the opportunity-cost floor is zero. At pressure, the
watermark is `max(0, min(eligible resident_value))`; admission requires a strict
improvement. Locked keys, pending transfers and required ancestors cannot be
victims. Only prefix leaves may be removed. Decisions use one observation time
and a virtual batch inventory; selected incoming blocks cannot be evicted by
later candidates in that batch. This is conservative greedy replacement, not a
globally optimal bundle/variable-size knapsack algorithm.

The single StoreController writer requests unlocked native deletion, then takes
a new snapshot after every singleton leaf deletion. An unconfirmed child stops
all remaining ancestor deletions, including when a native lookup races with the
proposal or a deletion fails. After all victims are confirmed, it re-plans.
Only a plan requiring no further deletion can be
submitted. Refused/failed/partial deletion never creates phantom free space.
The adapter's pending ledger charges submitted writes before another store can
be planned. Competing periodic L2 LRU eviction is disabled in this opt-in mode.
Unknown/stale profiles reject value-mode writes without eviction; they cannot
fall back to unbudgeted write-through. Default/native mode retains its fallback.
Failed writes do not restore deleted content; atomic prefix writes and rollback
of victim content are not promised. All L1 locks still belong to the controller.

### Comparison Contract

`--profiling-store-config` independently selects profiling admission while
`--eviction-policy ARC` retains native ARC L1 replacement. Its demand and L1
lifecycle observation must match the full policy's observation, including the
resident-count normalization. Duplicate shared and independent config is invalid.
Five arms use one final source/model/native build: stock ARC; ARC + original
admission; ARC + enhanced admission; original full policy; enhanced full policy.
Compare ARC vs full within each admission model, then original vs enhanced full.
Freeze traces and cyclic arm order before measurement; retain failed runs and
all evaluation requests. A friendly hot/scan result is not production evidence.
An experiment with zero capacity replacements does not prove watermark efficacy.

### Acceptance Mapping

| Property | Positive path | Negative path | Evidence |
| --- | --- | --- | --- |
| Resident-priced admission | replace cold leaf | reject lower net value, sunk-write accounting | policy entry-point tests |
| Owner capacity | completed + unique pending bytes | locked/pending/failed/partial writes | native adapter tests |
| Applied replacement | delete, refresh, submit | deletion refusal/error must release reads without store | real store-loop tests |
| Prefix safety | suffix-first replacement | retain required ancestors, unknown/stale history | policy tests |
| Algorithm isolation | ARC + same admission and lifecycle | duplicate/unsupported config fails construction | manager/config tests |
| TTFT comparison | complete independently restarted arms | incomplete/timeout/wrong arm/output mismatch rejected | benchmark runner/checker + raw GPU records |

Current implementation/validation results belong in the version-bound execution
record. This design text does not assert that unexecuted GPU criteria passed.

Continuous measured costs and calibrated reuse probabilities are also not supplied
by these changes. Existing numerical calibration drift stays unresolved unless
separately reproduced and fixed; illustrative inputs remain illustrative.

## Execution And Verification

- [x] Finish three frozen ARC/old-policy pairs and retain raw receipts/journals.
- [x] Independently audit comparisons, tracked-source/config parity and output equality.
- [x] Add RED tests: opt-in saturation, r-driven admission reversal, replica
  discount correctness, eviction ordering, invalid model/fraction rejection.
- [x] Implement the shared reuse-weight hook and local opt-in formula model.
- [x] Run affected CPU suites, default-behavior compatibility and type/lint checks.
- [x] Run available isolated GPU validation; distinguish historical source from
  current-dev source/native artifacts. No unbuilt-native compatibility claim.
- [x] Fresh-context independent review of immutable final code diff and relevant
  negative controls; preserve execution records. Do not mark ready automatically.

Acceptance is scoped functionality and honest comparison, not a required positive
TTFT result. ARC can match or beat the existing policy without invalidating the
experiment. Formula behavior tests do not prove performance superiority.

The first formula revision's source/CPU review closed at add4c52b. The dynamic
extension's runtime is 2efa3c89; e28c5be8 only broadens the real-FS test to CPU
pools without removing assertions. Independent review closed the prefix deletion
race/partial-failure finding with three-level controller/native-event probes.
Root and reviewer also passed actual native FS/controller pressure and retained
payload readback for both value models on CPU.

GPU-host runtime validation at 2efa3c89: 103 PASS, zero skips/failures, including
both real FS pressure/readback cases. This is native-controller/host-tensor
correctness, not model numerical or TTFT qualification. The extensions were built
in an isolated directory with coherent existing CUDA13.2 tools; C++/Rust/build
inputs are unchanged from d11ce959 through e28c5be8. All 738 deployed package
paths were subsequently matched to immutable e28c5be8 Git objects before the new
five-arm experiment. Partial transfer artifacts were not qualified.

The private execution record retains initial failures, review counterexamples,
source/native hashes and JUnit reports. Normal/live-idle timing, Sphinx, clean CI
and new model TTFT comparisons are NOT RUN at this documentation checkpoint.
The friendly workload may actively reject all cold L2 writes; that is not
unknown/stale fallback, but it also cannot establish watermark TTFT efficacy.
Historical ARC results remain historical. This is not PR readiness; these local
changes have not replaced the published PR head.
