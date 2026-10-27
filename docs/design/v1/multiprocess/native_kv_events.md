# Native CPU KV events

The `kv_event_native_bindings_v1` capability requires engine-prepared content
identities instead of reconstructed storage hashes. Enabling events
requires this capability and a compatible immutable `KVEventProfile`; it never
selects local store announcements or hash reconstruction on failure.

`IPCCacheServerKey` carries optional `event_profile` and `event_bindings` metadata
through the existing STORE RPC. A binding contains immutable native hashes,
parent, original token IDs, native block size and profile ID. Its tokens cover
exactly one complete LMCache chunk, including all C/B native subblocks. The
profile ID names the engine identity semantics, algorithm and seed contract.
MessagePack and gRPC preserve bytes and integer identities without conversions
between signed and unsigned representations. gRPC wraps each native identity in
a MessagePack scalar; its legacy LM chunk hash fields are reserved.
STORE keys retain packed big-endian uint32 `token_bytes`. Only each new event
payload slice is decoded; the scheduler keeps its incremental packed-token
tracker. Descriptors also carry `group_idx=0` and
`kv_cache_spec_kind="full_attention"`, including during cold-router replay.
Each batch explicitly carries native stream schema version 1; a missing version
decodes as zero and is rejected before any record is applied.

## Registration and commitment

Every worker supplies the optional profile in `register_kv_cache`, including
non-subscriber TP ranks and heartbeat re-registration. The actual candidate
layout is validated before changing shared registration state. Rejection closes
and reclaims the candidate context without unregistering existing ranks. The
elected subscriber also supplies model name, KV world size and profile in
`subscribe_kv_events`. Subscriptions reaching a replacement MP before its worker
registrations receive a transient error and retry the same source. The registered
layout supplies every required rank and object group. The configured positive
KV world size determines the complete rank set; validation targets TP1 and TP2.
Dense full attention and C divisible by B are validated;
salted keys, L2, GDS and incompatible profiles fail explicitly. Private
CPU-addressable Device-DAX uses the same L1 commitment notifications; the DRAM
acceptance matrix does not establish Device-DAX hardware validation. Engine
connectors reject unsupported model and topology configurations at startup.
vLLM uses the connector `validate_request` admission hook before the scheduler
owns the request. Event-enabled startup requires this engine API. Salted, LoRA,
multimodal, embedding, session, and resumable requests raise `VLLMValidationError`
at this request-local boundary. The hook reads immutable configuration only;
MultiConnector forwards validation to its children. Events-disabled requests
retain ordinary caching. Missing or insufficient hashes in a supported STORE
are errors, not a reason to omit bindings.
The vLLM admission rule follows its native hash semantics: an empty cache salt
is equivalent to no salt. This does not define SGLang's namespace semantics.
Native-event registration starts heartbeat on every rank before the elected
subscription waits for its initial schema-validated batch. Recovery reuses the
existing heartbeat and receiver rather than starting another service.

The event module keys bindings by `(model, world_size, cache_salt, chunk_hash)`.
`bind_store` validates conflicts synchronously. The existing EventBus joins those
bindings with `L1_WRITE_FINISHED` and `L1_WRITE_FINISHED_AND_READ_RESERVED` records.
A set of unique configured L1 tags identifies readable copies of each ObjectKey.
All required rank/group objects must have a committed copy before one Stored is
emitted. Losing the last copy of any required object emits one Removed with the
exact earlier native identities. Duplicate notifications and redundant copies do
not change this transition rule.

Bindings are retained by actual live objects and in-flight transfer IDs, not
request lifetime or a fixed LRU. Pending ownership uses the process-unique
transfer key; the request ID remains in event tracing metadata. A completion
callback first calls `finish_write_by_owner`, then enqueues the pending-binding
cleanup marker on the same EventBus as the resulting L1 notifications. GPU completion and
`MP_STORE_END` alone cannot prove that the host completion dispatcher has
finished. Failed writes release pending metadata without publishing a store;
partial committed chunks retain their binding while any required object lives.

## Replay and loss

Records own immutable descriptors, so cleanup or a later re-store cannot change
retained history. Records have monotonically increasing sequence IDs and a server
incarnation. Numeric gaps in a model-filtered stream are valid; `next_cursor`
tracks the global log. Log truncation reports explicit loss.

EventBus drop notification occurs after its preceding queue has drained. The
module ignores projection updates once it observes an unacknowledged drop, then
clears its rank/copy projection, pending bindings, live bindings and replay log,
then advances the replay floor. A later binding cannot resurrect pre-loss owners;
newly observed complete writes are required. This is conservative recovery, not
an authoritative snapshot. Warm objects whose records have fallen out of replay
may remain undiscovered until re-established with sufficient fresh evidence.

The common subscriber borrows the request client and owns only its stream/thread.
It invalidates CPU advertisements on reconnect/loss and replays the retained log.
The subscriber owns the receive buffer; each engine publication step builds its
output locally from those queued updates.
Engine publication retains its existing ownership, tracks advertised CPU IDs and
emits CPU-scoped removals; GPU placements are unaffected by CPU invalidation.
The publication ledger tracks each resident chunk's contribution to native blocks.
Sibling chunks may share a prefix shorter than C: only newly advertised block
runs are published, with their original token slices and preceding native parent.
Removing one chunk withdraws only blocks whose last contributing chunk is gone.
Duplicate chunk transitions and removals for chunks absent from retained replay
do not change another chunk's ownership. Loss clears both chunk contributions
and advertised blocks before replay. Lazy offload preserves native bindings in
chunk order when it coalesces contiguous STORE operations.

ZMQ returns validation and transient errors in separate response frames for
native-event registration, STORE, and subscription operations. gRPC maps the same
operations to `INVALID_ARGUMENT` and `UNAVAILABLE`, including its native server.
Incompatible registration fails explicitly while a subscription waiting for
re-registration can retry. Unrelated RPC error behavior is unchanged. Existing
typed clients bind optional registration defaults through `RpcSpec`; the raw
ZMQ payload count remains strict. Worker/server upgrades must be coordinated.

The bounded subscriber buffer replaces the extraction's receiver callback.
Only the engine's drain thread applies the contribution ledger and wraps native
events. Queue overflow always retains a reset before newer reliable data.
