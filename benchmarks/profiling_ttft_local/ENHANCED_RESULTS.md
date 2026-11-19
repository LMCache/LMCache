# Enhanced Local Policy: Validation Boundaries

The enhanced implementation adds opt-in at-least-one reuse probability,
recompute-cost discount, independent admission with native ARC/LRU, and
resident-priced L2 capacity replacement. These are mechanism changes, not a
qualified speedup. Historical LRU and ARC results in this directory retain their
original source versions and limitations.

## Historical Runtime Evidence

Native controller/FS validation executed at `2efa3c89`: 103 passed without skips,
including real FS pressure and retained-tensor readback. Test-only `e28c5be8`
also enables those FS assertions with CPU pools. These checks establish their
bounded controller/payload behavior, not model-generation equivalence, clean
native rebuilding, concurrent external-writer support or TTFT performance.

The five-arm model campaign used runtime `e28c5be8`, native ARC and independently
selected admission, three restart blocks and 900 responses. Its complete output
comparison passed, but strict shutdown qualification failed. Full-policy arms
performed no L2 stores or capacity replacements; their timings cannot establish
resident-value replacement efficacy.

## Main Capacity-Pressure Experiment: FAIL

Producer: `cb30010c3202fd3239666efc92b0c43ea956c600`.
Runtime: `e28c5be85776c229cbbd567573a4b22f17f78ae3`.
Supplemental acceptance: `661f707d`.
Raw archive SHA256:
`8e7f6dcc5f5962d835238ae378fdb3b42ecfb22ed5e1f5cb083059584d9b03d2`.

Nine jobs compare original/native capacity, probability/native capacity and
probability/value capacity across three cyclic restart blocks. All completed
with return code zero and no timeout: 756 responses, including 324 warmup and
432 measured. L1 was 0.5 GiB; declared L2 capacity was 336 MiB. Costs were fixed
illustrative inputs, not accepted measured calibration; recompute fraction was
zero. No actual partial-recomputation engine behavior is qualified.

| Property | Result | Boundary |
| --- | --- | --- |
| Complete jobs/responses | PASS | All nine jobs; completion alone is not acceptance |
| Real L2 activity | PASS | Stores, hits and observed file removals in all arms |
| Full output equivalence | FAIL | Six responses differ from the no-external-cache oracle; warmup is included |
| Strict shutdown logs | FAIL | Two jobs contain output-handler traceback/EngineDeadError despite exit zero |
| Accepted model-performance result | FAIL | Correctness and lifecycle gates failed; no enhanced TTFT gain is claimed |
| Current-dev model experiment | NOT RUN | Historical runtime evidence does not qualify a new integration head |

The value-capacity arms ended at the 336 MiB payload budget. Native periodic
capacity enforcement permitted larger transient usage. This observation concerns
managed payload, not a guarantee about total filesystem consumption.

## Diagnosis Is Not Repair

No-cache and ordinary native-APC controls each produced 252 matching responses.
A separate partial-prefix native-APC diagnostic, without any LMCache connector,
reproduced both alternate sequences from the failed experiment. LMCache is thus
not necessary for this symptom. The packets do not contain synchronized per-layer
KV transport checks, restored slot/position metadata or first-divergence logits;
LMCache payload correctness and a common numerical cause remain unproven.

The shutdown traceback has a source-supported cancellation-order hypothesis in
the pinned vLLM runtime. It is not an executed fix, nor permission to ignore
EngineDeadError. Real engine failures must remain visible. Neither excluding
warmup nor relaxing fatal-log checks repairs the failed experiment.

Next qualification requires bounded KV/metadata and same-prefix logit diagnosis,
shutdown reproduction and repair, followed by a complete fixed-recipe rerun.
Default, observe, active and live-idle overhead need separate measurements.
New-head local checks, independent review and hosted CI belong in the PR's
version-bound validation section; old results must not be relabeled as new runs.
