# Generation-fenced token-dropping lifecycle

> A model-agnostic control-plane contract for asynchronous decode-time cache
> edits. This module does not choose which tokens to drop or map logical spans
> to physical pages.

## Why the operation key needs a generation

Request IDs can be reused after abort, preemption, or rescheduling. A remote
completion carrying only `request_id` can therefore arrive during a different
lifecycle and mutate the wrong cache. Every operation instead carries:

```text
(request_id, request_generation, operation_id, drop_round, source_kv_revision)
```

`GenerationFencedDropLifecycle` retains a bounded set of old records as
diagnostic tombstones. Completions from an older generation become
`STALE_NOOP` even after pruning; duplicate live completions become
`DUPLICATE_NOOP`. `max_tombstones` defaults to 1024 and may be set to zero when
old-operation snapshots are not needed.

## Flow

```text
Q_CAPTURED
  -> PLAN_COMPUTING
  -> PLAN_READY
  -> PLAN_VALIDATED
  -> DEACTIVATION_SUBMITTED
  -> REMOTE_COMPLETE       source KV may now be reused
  -> LOCAL_VISIBLE         edit effect is applied exactly once
  -> PLAN_CONSUMED
```

Any active phase may instead end in `ABORTED`, `STALE`, `FAILED`, or
`RECOMPUTE_REQUIRED`. Topology, policy, source-KV, and accepted-sequence-length
mismatches fail closed to recomputation.

## Example

```python
from lmcache.sdk.drop_lifecycle import (
    DropOperationCompletion,
    DropOperationState,
    GenerationFencedDropLifecycle,
)

lifecycle = GenerationFencedDropLifecycle(
    request_id="chat-42",
    request_generation=3,
    topology_fingerprint="hybrid-c4-c128-v1",
    policy_revision="drop-policy-v7",
)
key = lifecycle.begin_operation(
    drop_round=2,
    source_kv_revision=11,
    accepted_seq_len=4096,
)
for state in (
    DropOperationState.PLAN_COMPUTING,
    DropOperationState.PLAN_READY,
    DropOperationState.PLAN_VALIDATED,
    DropOperationState.DEACTIVATION_SUBMITTED,
):
    lifecycle.advance(key, state)

completion = DropOperationCompletion(
    key=key,
    topology_fingerprint="hybrid-c4-c128-v1",
    policy_revision="drop-policy-v7",
    accepted_seq_len=4096,
)
lifecycle.record_remote_completion(completion)
lifecycle.mark_local_visible(completion)
lifecycle.consume(key)
```

When the serving engine reuses `chat-42`, advance the generation before
starting new work:

```python
lifecycle.advance_generation(
    4,
    topology_fingerprint="hybrid-c4-c128-v1",
    policy_revision="drop-policy-v8",
)
```

A late generation-3 completion is now a side-effect-free stale event. Calling
`abort()` repeatedly is also safe: only the first call changes active records.

## Integration boundary

The tracker owns identity, ordering, fencing, deduplication, and public metrics.
The serving adapter still owns Q capture, plan computation, physical cache
operations, and recomputation. In particular, the backend must echo the exact
`DropOperationCompletion` metadata submitted with the operation; reconstructing
completion fields from current request state defeats the generation fence.
