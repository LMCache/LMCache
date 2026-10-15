# Historical Admission Opportunity Experiment

Refs [#5782](https://github.com/LMCache/LMCache/issues/5782). This delivers a
bounded offline opportunity screen, not an online cost-aware policy.

## Contract

Use the existing `LRUCacheFast` implementation and prefix-hit definition.
Replay requests serially in recorded order; equal timestamps preserve file
order. A request reads the pre-write cache, then touches/inserts blocks in
request order. Completed writes become available to the next request.
There is one cache tier and no GPU automatic prefix cache (APC).

Modes are no-offload, write-all LRU and historical prefix admission. All
cache-enabled modes use the same capacity in full 512-token blocks. The
historical selector keeps at most `history_keys` last-observation records,
independent of residency. A block qualifies only if an earlier request
observed it within `max_request_gap` requests. Selection stops at the first
unqualified block, so it cannot store an isolated later block as useful work.
The current request enters history only after selection.

This gap counts requests, not intervening distinct keys, wall-clock delay or
a calibrated reuse probability. It is a deliberately small historical
baseline. This experiment does not port the old profiler's uncalibrated
probability/co-occurrence machinery into production.

An unlimited-capacity, write-all prefix cache supplies an optimistic
reuse-work ceiling. It is NOT a finite-capacity optimum, a goodput bound,
or a model of current lazy offload. No future information enters the
historical selector. Earlier requests warm both caches; `evaluation_start`
excludes them from reported totals. Freeze settings before examining an
evaluation partition; repeated tuning on that partition is not held-out evidence.

## Inputs And Failure Behavior

The input is Mooncake FAST25 JSONL, not LMCache lookup JSONL. Require finite
nondecreasing timestamps, positive integer input length, nonnegative integer
output length and exactly `ceil(input_length / 512)` integer hash identifiers.
Only full input blocks participate; a partial final block is never split or
counted as a hit. Output tokens are not cached. Every repeated identifier
must have identical ancestry, including partial blocks.

The caller provides a SHA-256 acquired from a pinned input manifest. The
digest covers exactly the bytes parsed in the same pass and is verified
before returning a report. Reject malformed/empty/truncated/drifted input,
inconsistent ancestry, invalid settings and an empty evaluation window.
The CLI exits nonzero without a success JSON report. Trace limits are
100,000 requests, 4,096 blocks per request and 2,000,000 distinct identifiers;
crossing a limit fails rather than silently truncating. Online history is
bounded separately. Offline ancestry and ceiling state are trace-size-bounded,
not production-resident metadata.

Suggested pinned source:
`kvcache-ai/Mooncake@3aa0f716008c0cac25ffc82ef234d6b0a59e9c2e`,
`FAST25-release/traces/{conversation_trace,toolagent_trace}.jsonl`.
Verify downloaded bytes against the repository blob/size, then retain SHA-256
and all settings with the JSON report. A checksum of an unfinished download
is not a valid source manifest.

## Validation And Scope

Acceptance tests invoke the actual executable module and check exit status,
JSON qualification, totals and diagnostics. Negative cases include schema
types (including bool-as-int), invalid JSON, blank/empty input, partial input,
digest drift, ancestry conflict and an empty evaluation partition. Exhaustive
small request combinations compare baseline work to the existing simulator
and check that neither modeled policy exceeds the unlimited reuse ceiling.
The history test exercises cold/current-request ordering, prefix closure,
age expiry, forgotten keys and a stream much longer than the history bound.

`status=PASS` means a valid replay, not an admission-benefit acceptance gate.
Reports list current lazy scheduling, GPU APC, asynchronous completion,
transfer/recompute costs and TTFT/ITL/goodput as unmodeled. Insertion counts
are not D2H traffic, and lower writes alone do not establish a serving benefit.
Actual cost calibration, live observation integration and eager/lazy GPU cap
experiments remain separate qualifications before an online policy is enabled.
