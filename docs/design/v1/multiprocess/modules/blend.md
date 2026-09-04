# Blend server (`lmcache/v1/multiprocess/modules/blend/`)

The blend module implements **non-prefix KV reuse** (CacheBlend) on the
LMCache multiprocess server: it fingerprints stored chunks, matches them at
any position in a later request, prefetches the matched KV, and scatters it
into the request's paged blocks — re-RoPE'd to the new positions — so the
client recomputes only the holes.

The vLLM-process side (connector, forward pass, blending strategy) lives in
the CacheBlend plugin; the only coupling is the wire contract below.

## Package layout

```text
blend/
├── module.py            BlendModule: mixin composition, __init__ (all state),
│                        get_handlers, protocol handshake, liveness, close
├── lookup.py            LookupMixin — CB_UNIFIED_LOOKUP FSM: prefix leg,
│                        local fingerprint match, coordinator leg, sparse leg
├── registration.py      RegistrationMixin — CB_(UN)REGISTER_ROPE, the STORE
│                        fingerprint hook, async fingerprint drainer
├── retrieve.py          RetrieveMixin — CB_RETRIEVE_PRE_COMPUTED: planning
│                        section (invariant specs + flat int64 work tables)
│                        first, then the handler
├── scatter_fallback.py  ScatterFallbackMixin — Python wave loop (pre-plan
│                        cuda_ops builds and inputs the planner declines)
├── matcher.py           pure: BlendTokenRangeMatcher (fingerprint index)
├── rope.py              pure: _CBRopeState, rope geometry rules
└── read_set.py          pure: per-leg object-group read sets and key
                         expansion
```

The pure modules have no server context, streams, or event bus — they are
CPU-testable by import. All mutable state is created in
`BlendModule.__init__` (`module.py`); mixin methods share it on `self`, and
each mixin declares the attributes it reads in a `TYPE_CHECKING` block.

## Wire contract

Request ids are pinned and append-only (`protocols/base.py`); payload and
response shapes are frozen per `BLEND_PROTOCOL_VERSION`
(`protocols/blend.py`) — an incompatible shape change means a **new request
name**, never a changed one.

| RPC | Payload → Response | Semantics |
|---|---|---|
| `CB_REGISTER_ROPE` | `(instance_id, cos_sin_ipc[], head_size, is_neox, group_to_cache[], group_rot[][])` → `None` | idempotent; strips baked mscale; MLA must declare its rope window; zero caches = NoPE |
| `CB_UNREGISTER_ROPE` | `(instance_id)` → `None` | KV cache stays registered |
| `CB_UNIFIED_LOOKUP` | `(key, tp_size)` → `CBUnifiedLookupResult \| None` | submit-once / poll-on-recall; `None` = defer, client re-issues |
| `CB_RETRIEVE_PRE_COMPUTED` | `(key, matches[], gpu_block_ids[][], instance_id, event_ipc)` → `(event_ipc, scatter_ran)` | event is server-recorded; may be called more than once per request |
| `CB_PROTOCOL_HANDSHAKE` | `(client_version)` → `(server_version, compatible)` | client-gated by `cb.handshake`, default off |

`STORE` is shadowed: the compositor registers the blend module last, so its
`store` wins and wraps `LMCacheDrivenTransfer.store` with fingerprint
registration.

## Unified lookup (submit-once, poll-on-recall)

The handler never holds a worker thread across L2→L1 loads: the first call
submits work and returns `None`; each later call polls, returning `None`
until every leg is resident.

```mermaid
stateDiagram-v2
  [*] --> PrefixPending: 1st call — submit prefix prefetch + local match / coordinator query
  PrefixPending --> PrefixPending: not resident → None
  PrefixPending --> Reconcile: prefix_chunks (+ retained set if SEGMENTED_PREFIX)
  Reconcile --> CoordPending: coordinator PENDING, before deadline → None
  CoordPending --> Reconcile: resolved or deadline
  Reconcile --> SparsePending: candidates cur_st ≥ prefix → submit_prefetch_task(SPARSE) once
  SparsePending --> SparsePending: not resident → None
  SparsePending --> Done: classify · strikes · overlap dedup · stash obj_keys
  Done --> [*]: CBUnifiedLookupResult
```

The two legs trim opposite ways:

| | Prefix leg (`PREFIX` / `SEGMENTED_PREFIX`) | Non-prefix leg (`SPARSE`) |
|---|---|---|
| Keys | contiguous chunk-hash chain from 0, over `prefix_gids` (attention + recurrent) | matched chunks anywhere, over blend `gids` (attention + aux) |
| Result | leading-ones count via the window-aware fold — truncate at the first gap; `SEGMENTED_PREFIX` additionally retains fully-loaded post-gap chunks (off for recurrent registrations) | keep every chunk whose **entire (read-group × rank) key set** loaded; no contiguity |
| Why | vLLM consumes the prefix as one `num_computed_tokens`; recurrent state needs unbroken history | chunks relocate independently; the forward recomputes the holes |

A chunk missing **any** rank's or **any** read group's key is dropped whole
and takes a stale strike (evicted from the matcher at the strike threshold);
the rest of the request proceeds. The found set's object keys are stashed in
`Session.extras` for the retrieve; whatever no retrieve consumes is released
by the session-destroy listener (the sparse-lock-leak fix, #4852).

## Retrieve (plan-then-execute, all-or-nothing)

`cb_retrieve_pre_computed` fills temp slots from L1 (H2D), K-only re-RoPEs
the shifted subset, and scatters **per token** into the paged KV — so
non-block-aligned matches and partial vLLM blocks shared with recomputed
tokens are written correctly.

The fast path builds one flat native plan (invariant specs cached per GPU
context, stamped with the request's slot mappings; work encoded as numpy
int64 tables) and enqueues everything in a single `cuda_ops` call. The
Python wave loop in `scatter_fallback.py` covers builds that predate the op
and inputs the planner declines.

The scatter is **all-or-nothing** — never partial. Zero-work returns are
reported with a fixed reason code on `CB_RETRIEVE_NOOP`:

| Reason | `scatter_ran` | When | Client effect |
|---|---|---|---|
| (success) | `True` | every matched range scattered | blend forward proceeds |
| `already_applied` | `True` | repeat call, same destination blocks | no-op by design |
| `awaiting_full_alloc` | `True` (no publish) | **all** matches beyond the allocated slots | defer to vLLM's full-alloc follow-up call; locks stay held |
| `partial_alloc` | `False` | **some** matches beyond the allocated slots while others are forwarded this step | client degrades the request to full recompute (TP-consensus, no raise) |
| `no_object_keys` | `True` | nothing to read | silent full recompute |
| read/scatter failure | `False` | prefetched objects unavailable, or an exception mid-scatter | client degrades the request |

Invariant: `scatter_ran=True` implies every matched row the client forwards
this step is backed by scattered KV. Every return path exports a **freshly
recorded server event** — echoing the caller's own IPC handle back makes the
worker re-import it (CUDA "invalid device context").

Repeat calls are keyed by the destination blocks each range writes into
(bounded LRU per `(request, worker)`): block-table growth keeps a range
applied, a reassigned destination re-scatters.

## Locks

Sparse-prefetch read locks follow one rule: exactly one owner releases each
reservation. The retrieve releases applied ranges stream-ordered after the
scatter, releases lookup-stash orphans it will never read, and leaves
beyond-slot-bound ranges locked for the follow-up call; anything never
consumed by a retrieve is released on session destruction
(`_release_unretrieved_locks`). The L2 regression harness gates changes
here.

## Observability

Every `cb.*` span/event is published from the code path it measures:
`CB_REQUEST_*`, `CB_LOOKUP_*`, `CB_PREFIX_LOOKUP_*`,
`CB_FINGERPRINT_MATCH_*`, `CB_COORDINATOR_MATCH_*`, `CB_SPARSE_PREFETCH_*`,
`CB_RETRIEVE_*`, `CB_SCATTER_*`, `CB_RETRIEVE_NOOP`,
`CB_FINGERPRINTS_REGISTERED`, `CB_CHUNKS_EVICTED`. Retrieve/scatter events
stamp `worker_id` (a source-level test enforces this — at TP>1 the metrics
subscriber pairs START/END per rank). `cb.request` is closed by whichever
path finishes the request: the lookup when there is nothing to retrieve,
otherwise the last retrieve.

## Test map

| Concern | Tests |
|---|---|
| Matcher / dedup-content | `tests/v1/multiprocess/test_blend_matcher.py` |
| Retrieve planner (flat plan) | `tests/v1/multiprocess/test_blend_retrieve.py` |
| Rope batching, lock lifecycle, fingerprints, read sets | `tests/v1/multiprocess/test_blend_load_store_opts.py` |
| Event emission / spans | `tests/v1/multiprocess/test_blend_observability.py` |
| Rope geometry (fused / MLA) | `tests/v1/multiprocess/test_cb_fused_geometry.py`, `test_cb_mla_rope.py` |
| Wire contract | `tests/v1/multiprocess/test_protocols.py` (frozen ids, handshake) |
