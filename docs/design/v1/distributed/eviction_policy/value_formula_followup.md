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

StorePolicy receives configuration-only L2 descriptors; adapter usage snapshots
and key listings are separate, and there is no atomic reserve/replace decision
shared with native L2 eviction. Bounded completion history is not an authoritative
inventory. This change must NOT claim dynamic L2 opportunity-cost enforcement.
It retains the zero opportunity-cost floor, explicit one-time write charge and
native L2 eviction. Full capacity-aware admission requires a separate owner-level
reservation/eviction contract. L1 ordering chooses low-value eligible residents,
but is not pre-allocation L1 admission.

Continuous measured costs and calibrated reuse probabilities are also not supplied
by these changes. Existing numerical calibration drift stays unresolved unless
separately reproduced and fixed; illustrative inputs remain illustrative.

## Execution And Verification

- [ ] Finish three frozen ARC/old-policy pairs and retain raw receipts/journals.
- [ ] Independently audit comparisons, source/config parity and output equality.
- [ ] Add RED tests: opt-in saturation, r-driven admission reversal, replica
  discount correctness, eviction ordering, invalid model/fraction rejection.
- [ ] Implement the shared reuse-weight hook and local opt-in formula model.
- [ ] Run affected CPU suites, default-behavior compatibility and type/lint checks.
- [ ] Run available isolated GPU validation; distinguish historical source from
  current-dev source/native artifacts. No unbuilt-native compatibility claim.
- [ ] Fresh-context independent review of immutable final diff and relevant
  negative controls; preserve execution records. Do not mark ready automatically.

Acceptance is scoped functionality and honest comparison, not a required positive
TTFT result. ARC can match or beat the existing policy without invalidating the
experiment. Formula behavior tests do not prove performance superiority.
