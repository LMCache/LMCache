# Topology-aware token-selection plans

Token-selection policy should describe semantic retention decisions, not CUDA
page indices, padded strides, rank-local addresses, or model-specific block
tables. `lmcache.sdk.token_selection` separates those layers:

- `TokenSelectionPlan` carries request/revision fences and logical group spans.
- `CacheTopologyDescriptor` describes compression, alignment, sharding, active
  windows, and cache-family membership.
- `expand_plan()` first validates every fence and family invariant, then emits
  full-block `PhysicalGroupOperation` values.

This hides three kinds of adapter complexity from policy authors: translating
logical token spans through compression and alignment into physical entry/byte
ranges, atomically validating every member of a hybrid cache family, and
rejecting stale request/topology/policy revisions before any destructive range
is exposed. The serving adapter still owns the actual cache mutation and must
handle explicit residual spans; the abstraction does not conceal that side
effect or silently widen a partial physical block.

## Fail-closed boundaries

Expansion returns no operations when request generation, decode round,
accepted sequence length, source KV revision, topology fingerprint, or policy
revision differs from live state. The same applies to unknown groups, expired
decisions, selections past the accepted-token boundary, or incomplete cache
families. In each case `validation.recompute_required` is true.

Plan serialization uses a total ordering over selection content. Even malformed
duplicate-group input therefore has one stable digest before validation rejects
it, independent of transport tuple order.

Hybrid groups use symmetric `sibling_group_ids`, and every member must declare
the same complete family. Overlapping or partially connected sibling graphs are
rejected when the topology is built. Every selected family member must be
present with the same action, logical span, and validity deadline. A partial
hit is therefore an all-family miss and must be recomputed.

## Compression and residuals

Only complete logical compression blocks become physical entry ranges. For a
C4 group, `[3, 10)` expands as:

```text
full logical block: [4, 8)
physical entries:   [1, 2)
residual spans:     [3, 4), [8, 10)
```

This prevents an invalidate or demotion from silently widening to tokens that
the semantic plan did not select. The adapter must handle `residual_spans`
explicitly, typically by preserving or recomputing them. A span containing no
complete compressed block emits an empty physical range plus the full logical
span as residual work.

## Example

```python
from lmcache.sdk.token_selection import (
    CacheGroupGeometry,
    CacheSemanticKind,
    CacheTopologyDescriptor,
    GroupSelection,
    LogicalSpan,
    RetentionAction,
    TokenSelectionPlan,
    expand_plan,
)

main = CacheGroupGeometry(
    group_id="compressed-main",
    semantic_kind=CacheSemanticKind.COMPRESSED_DENSE,
    logical_tokens_per_block=4,
    physical_entries_per_block=1,
    compression_ratio=4,
    rank_sharding="tp-contiguous",
    page_stride_bytes=256,
    alignment_bytes=128,
    sibling_group_ids=("indexer",),
)
indexer = CacheGroupGeometry(
    group_id="indexer",
    semantic_kind=CacheSemanticKind.INDEXER,
    logical_tokens_per_block=1,
    physical_entries_per_block=1,
    compression_ratio=1,
    rank_sharding="tp-contiguous",
    page_stride_bytes=256,
    alignment_bytes=128,
    sibling_group_ids=("compressed-main",),
)
topology = CacheTopologyDescriptor.create(
    model_architecture="HybridForCausalLM",
    backend_name="paged-backend",
    topology_version="v1",
    groups=(main, indexer),
)

span = LogicalSpan(0, 256)
plan = TokenSelectionPlan.create(
    request_id="chat-42",
    request_generation=3,
    decode_round=2,
    accepted_seq_len=256,
    source_kv_revision=11,
    topology_fingerprint=topology.fingerprint,
    policy_revision="drop-policy-v7",
    groups=(
        GroupSelection(
            "compressed-main",
            CacheSemanticKind.COMPRESSED_DENSE,
            span,
            ("indexer",),
            RetentionAction.INVALIDATE,
        ),
        GroupSelection(
            "indexer",
            CacheSemanticKind.INDEXER,
            span,
            ("compressed-main",),
            RetentionAction.INVALIDATE,
        ),
    ),
)
expansion = expand_plan(
    plan,
    topology,
    request_id="chat-42",
    request_generation=3,
    decode_round=2,
    accepted_seq_len=256,
    source_kv_revision=11,
    policy_revision="drop-policy-v7",
    current_step=2,
)
assert expansion.validation.valid
```

For a live LMCache/vLLM path in which the expanded physical range drives an
actual `LMCacheRequestStream.modify_kv()` operation, see
[`examples/token_dropping/topology_aware_token_dropping.py`](../../../examples/token_dropping/topology_aware_token_dropping.py).
