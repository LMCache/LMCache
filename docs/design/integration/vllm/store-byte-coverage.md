# Store-Time Byte-Coverage Check

## Summary

A fail-closed store-time invariant that catches the "silently stores 1 of N
kernel pages" bug family (e.g. [#4701], and the MLA/DSA arm fixed by [#5971]):
a registration-geometry bug resizes each stored chunk to a fraction of its
true KV payload, bookkeeping stays internally consistent, hit rates stay
high, and the corruption only surfaces downstream as wrong answers after a
restore. The check derives the expected chunk bytes **only from the serving
engine's KV cache spec**, compares them against the bytes LMCache actually
reserves when staging a chunk, and refuses to persist the store on mismatch
— turning the whole family from silent corruption into a loud, per-operation
error.

## Design invariants

Three properties keep the check honest (from the cross-family evidence in
[#4701]):

1. **Independent derivation.** The expected byte count never reads LMCache's
   own registration geometry (`kv_layer_groups` shapes, `PageBufferShapeDesc`
   slot math) — that geometry is exactly what was wrong in every known
   instance. It is computed at registration time, in the single vLLM→LMCache
   conversion entry point (`create_engine_group_infos_from_vllm`), from the
   engine leaf specs only: `expected_bytes_per_token = Σ_layers
   (page_size_bytes / block_size)`, DCP scaling aside. Both sides of the
   comparison therefore share no intermediate arithmetic; they only meet at
   the comparison point.
2. **Checked at a lossless boundary.** The comparison happens on the host
   **memory object reserved for the chunk**, before any device transfer and
   before any (encoded/serialized) L2 persistence — never on serialized disk
   sizes, which are not byte-comparable to raw KV.
3. **Reservation size is not transfer completeness.** A matching byte count
   proves the reserved object had the engine-spec byte capacity for the
   chunk — an independently derived reservation-size invariant. It does
   not prove the device transfer populated every page (missing or
   duplicated writes inside a right-sized allocation can still pass), nor
   page identity or content (a right-sized object can still be
   zero-filled, mis-ordered, or torn). The check is the cheap always-on
   gate at the base of the test pyramid; restart/exact-answer restore
   tests remain the transfer-and-restore correctness layer on top.

## Mechanics

- At registration, each `EngineGroupInfo` carries
  `expected_bytes_per_token` (defaulted `msgspec` field, wire-compatible:
  old daemons ignore it, new daemons see `None` from old connectors). The
  grpc transport mirrors the struct field-for-field
  (`common.proto` `EngineGroupInfo`, presence-typed `optional int64`
  field 7): its structural codec enforces exact python↔proto field
  parity, so the two definitions must move together.
  `KVLayerGroupsManager` stamps it onto each `KernelGroupInfo`, and
  `expected_object_group_bytes_per_chunk(og, chunk)` sums the rates over an
  object group's member kernel groups × the chunk size.
- At store time (`LMCacheDrivenTransferModule.store`), right after the
  storage manager reserves a chunk's objects, `verify_store_byte_coverage`
  compares the reserved object size against that expectation. A mismatch
  raises inside the existing per-request `try`, so the already fail-closed
  commit gate skips the whole store: nothing is persisted, the server
  survives, and the worker logs the failure for that request only. The
  store-outcome plumbing (`(event, bool)` over the message queue) is
  unchanged.
- Multiple planes within one registration are safe by construction: the
  rate is summed per **member layer of each transfer identity group**, so a
  main-cache plane and a key-only/indexer plane each carry their own rate,
  and the object-group sum reassembles the full payload.

## No-op scope (fail-open for underivable, fail-closed for derived)

The check only fires when it can compute a trustworthy expectation. It
no-ops (`expected_bytes_per_token is None`) for:

- recurrent-state groups (Mamba/GDN pages hold state snapshots, not
  per-token KV),
- sliding-window groups (cross-chunk window staging, unless `full_sw_kv`),
- DCP topologies (`dcp_size > 1`; per-rank page semantics are not derivable
  from the spec alone),
- dim-0-padded pools (`block_stride_elems` does not tile the tight page —
  a numel-based comparison would be unreliable),
- TurboQuant-style specs (`tq_slot_size > 0` alters page semantics), and
  any spec missing a divisible `page_size_bytes / block_size`.

A missing expectation is a no-op, never an error — an exotic layout must
not break. A *derived* mismatch is an error: at that point the two
independently computed numbers disagree, which is precisely the corruption
signature.

## Kill switch

`LMCACHE_STORE_BYTE_COVERAGE_CHECK` — default **enabled**; set it to
`0`/`false`/`no`/`off`/empty to disable (read once at module init, so the
per-store cost is a flag test). The check itself is O(groups) arithmetic
per store with no data scans.

## Non-goals

- Content validation (checksums/CRC of stored pages) — a separate, more
  expensive track; this check only proves coverage.
- Retrieval-side checks — a store that passes here cannot have committed
  a partial chunk, so retrieve trusts the committed layout.
- The legacy in-process connector path: it registers bare tensors with no
  engine spec, so no independent expectation exists there. The modern MP
  (`LMCacheMPConnector` + LMCache server) path — where every known family
  instance ran — is covered.

## Follow-ups

- Registration-time operator visibility: expose checked-versus-skipped
  object groups with the no-op reason (a log line and/or counter at
  registration), so operators can tell "protected" from "not checked" on
  a running cluster. Kept out of this PR to hold its footprint; the
  no-op reason data (`expected_bytes_per_token is None` + which no-op
  branch fired) is already computed at registration.

[#4701]: https://github.com/LMCache/LMCache/issues/4701
[#5971]: https://github.com/LMCache/LMCache/pull/5971
