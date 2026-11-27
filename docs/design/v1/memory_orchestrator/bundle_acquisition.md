# Bundle and Acquisition services

## Scope

These services extend the memory orchestration control plane for PD handoff.
They are metadata services: they do not allocate memory or touch KV bytes.

- **Bundle Service** records one immutable checkpoint requirement: the exact
  prefix boundary, chunk size, required component rows, and component windows.
- **Acquisition Service** tracks one delivery attempt to one destination. It
  validates grants, tracks progress and readiness, and coordinates safe cleanup.

The target is MP-mode PD handoff. An acquisition may use private DRAM, shared
L1, or both. The services do not require a shared region and must support zero
managed shared regions.

This milestone defines interfaces and lifecycle semantics only. It is MP-only:
in-process interfaces, configuration, and runtime behavior remain unchanged. A
compatible PD deployment and data path are prerequisites; their implementation
is outside this document.

## Why separate services

Region/Object tracks object readiness inside a region. It does not know whether
a set of objects forms one restorable checkpoint:

```text
Boundary                 1024       2048
FULL KV                  yes        yes
GDN checkpoint           yes        no
Restorable @1024         YES
Restorable @2048         NO
```

Object-level `VALID` therefore cannot decide PD readiness. A bundle expresses
the exact checkpoint, and an acquisition verifies that all required components
cover that boundary at one destination.

Failure has a second distinction: reporting an operation as failed does not
prove that a transfer, a consumer, or another reader has stopped using a buffer.
Failure reporting and cleanup completion are separate states.

Each destination progresses independently. One failed acquisition does not
release another destination's grants or block destinations that are ready.

```text
Memory orchestrator (metadata only)
|
+-- Region / Object Service: allocation, publication, leases, generations
+-- Bundle Service: immutable checkpoint requirements
+-- Acquisition Service: grants, progress, readiness, cleanup
+-- Existing owners: validate and release grants through their normal paths
```

Buffer owners remain authoritative. Shared regions own shared leases; existing
L1 managers own private allocations and locks. The orchestrator only records
owner-confirmed state.

## Bundle Service

A bundle is an immutable requirement, not a snapshot of current availability.
Its ID is the digest of the canonical manifest, so a caller-provided label
cannot alias two checkpoints. Repeated creation of the same manifest is
idempotent; corrupted or mismatched content fails as a conflict.

| Field | Meaning |
| --- | --- |
| `bundle_id` | Content-addressed identity of the canonical manifest. |
| `prefix_tokens` | Exact token boundary represented by this checkpoint. |
| `chunk_size_tokens` | Token span of one object chunk. The boundary must be chunk-aligned. |
| `components` | Required object-group and recurrent-state components. |

| Component field | Meaning |
| --- | --- |
| `component_id` | Manifest-local reference used by grants and progress. |
| `object_group_id` | Model object-group identity. |
| `world_size` | Number of kv-rank rows in this component. |
| `rows` | One chunk-ordered `ObjectKey` row per kv rank. |
| `window_chunks` | Cross-chunk dependency: `-1` full attention, `>= 1` sliding window. Zero is invalid. |
| `component_tag` | Stable semantic identity, such as `full_attention_kv` or `gdn_recurrent_state`. |

An `ObjectKey` identifies one chunk for one object group and rank, so coverage
uses ordered key rows rather than one key per shard. All rows must have the
manifest's exact chunk count, and keys in a row must agree on model, salt,
object group, and kv rank. Recurrent state uses `component_tag`; it is not
inferred from shapes.

`CreateBundle` validates only the manifest. It does not claim that the objects
are currently present. Current availability is checked during each acquisition
while grants protect the rows.

Coverage reuses `fold_grouped`: one presence bitmap per manifest row, paired
with that component's window. Bit `j` represents a servable prefix of `j + 1`
chunks, so a checkpoint of `N` chunks checks bit `N - 1`. The exact boundary
must pass; a different restorable boundary requires a different bundle.

| RPC | Does |
| --- | --- |
| `CreateBundle` | Validate and store an immutable manifest; return `CREATED`, `ALREADY_EXISTS`, `CONFLICT`, or `INVALID_MANIFEST`. |
| `DescribeBundle` | Return the manifest for a bundle ID, or `NOT_FOUND`. |

A bundle remains referenced while an acquisition, including a failed/quarantined
record, is cleaning up. Deletion and bounded-retention GC are follow-ups.

## Acquisition Service

An acquisition is one attempt to deliver one bundle to one destination. It can
reference grants from several owners and regions.

| Acquisition field | Meaning |
| --- | --- |
| `acquisition_id` | Globally unique operation identity. |
| `bundle_id` | Checkpoint being acquired. |
| `destination_id` | Destination participant identity. |
| `caller` | Acquisition creator identity and incarnation. |

An acquisition has no operation-level region or single participant incarnation.
Each grant carries its own owner incarnation; shared grants additionally carry
region identity and epoch. The source and destination pairing is a deployment
prerequisite and is not selected or validated by Acquisition. A request is
idempotent by caller and `request_id`; reusing an acquisition ID or request ID
for different content is a conflict.

### Grants

A grant is a reference to a reservation already held by a buffer owner.
`StartAcquisition` asks the owner to validate it and return the protected rows
used for coverage. The service records that response; it does not invent a
second lock table.

| Grant field | Meaning |
| --- | --- |
| `grant_id` | Owner-generated identity of this reservation. |
| `role` | `SOURCE` or `DESTINATION`. |
| `owner` | Owner identity and owner incarnation. |
| `placement` | Shared-region grant or private-L1 grant. |
| `components` | Manifest rows and chunk ranges protected by the grant. |

A shared grant carries region ID, region epoch, allocation generation, and read
lease tokens. A private grant carries the L1 manager identity, allocation
generation, and owner reservation tokens. Region identity is never replaced by
a dummy value for private DRAM.

The union of grants must cover every required source row and every required
destination row. A direct shared-memory read may have only a source grant.
Private and shared grants may be mixed in one acquisition. Grant validation
checks ownership, generations, and protected rows.

If several acquisitions use the same source allocation, the owner must issue
independently releasable grants or maintain its own reference count. One
acquisition completing does not release another acquisition's source access.

### RPCs

| RPC | Does |
| --- | --- |
| `StartAcquisition` | Validate identity and grants with their owners, verify coverage, and create the operation. |
| `AcknowledgeProgress` | Record owner-confirmed progress for one grant. |
| `QueryAcquisitionStatus` | Return operation outcome, cleanup state, and per-grant progress. |
| `ReleaseAcquisition` | Start cleanup; does not synchronously declare buffers safe. |

Owner validation happens before the acquisition becomes active. If a later
validation step fails, the service either releases grants already touched or
retains a queryable failed record whose cleanup is `BLOCKED`. It never reports
a clean failure if rollback cannot be confirmed.

### Progress

Progress is per owner, grant, component, and chunk. An acknowledgement must
match the recorded owner incarnation, role, placement identity, region epoch
where applicable, and allocation generation. Each grant has a monotonic
sequence; timestamps are diagnostic only.

| Phase | Meaning |
| --- | --- |
| `TRANSFER_STARTED` | Data movement was submitted. |
| `TRANSFER_COMPLETED` | Data movement finished; this alone is not readiness. |
| `DATA_AVAILABLE` | Shared-memory owner confirms the leased object is published and visible. Used instead of transfer completion for a direct shared read. |
| `IMPORT_COMPLETED` | Destination-side consumption confirmation completed. A no-op confirmation must be explicit. |
| `ACCESS_STOPPED` | Owner confirms no transfer, consumer, or other reader can still access the buffers. |
| `RELEASED` | Owner confirms release through its existing ownership path. |
| `FAILED` | A data-path operation failed; this is not permission to release. |

Duplicate acknowledgements are idempotent. Out-of-order phases remain per grant
and cannot overwrite a newer sequence. Stale or unauthorized acknowledgements
are rejected without changing state.

### Readiness

An acquisition becomes `READY` only when:

- the operation has not failed or been cancelled;
- all required manifested rows have current source grants;
- separate destination storage has current destination grants;
- exact bundle coverage passes;
- each remote path has transfer completion, and each direct shared read has data
  availability;
- destination consumption confirmation is complete.

Readiness is per acquisition. There is no 1P:ND barrier.

### Outcome and cleanup

Operation outcome and cleanup state are independent:

```text
Outcome: PENDING | COPYING | READY | FAILED | CANCELLED
Cleanup: NONE | PENDING | RELEASING | DONE | BLOCKED
```

`FAILED` means the operation cannot be used; `cleanup=DONE` means all required
buffers are safe to reuse. A timeout can therefore produce
`FAILED / cleanup=PENDING` followed by `FAILED / cleanup=RELEASING`.

Cleanup completes only after every required owner confirms `RELEASED`.
`RELEASED` follows `ACCESS_STOPPED`, unless an owner release operation atomically
proves both. Transfer completion alone is insufficient.

If an owner is unreachable or release is ambiguous, cleanup becomes `BLOCKED`.
The buffers remain quarantined or offline until an operator or future fencing
protocol resolves the state. A late acknowledgement may advance cleanup but
cannot make a failed acquisition ready.

`ReleaseAcquisition` starts cleanup after readiness, failure, or while copying.
For a copy in progress, the transfer must first be stopped or awaited. An
acquisition ID is never reused for a new operation.

### Re-registration and restart

A new participant incarnation fences its old operation progress but does not
prove that old data movement stopped. Old acquisitions move to cleanup `PENDING`
or `BLOCKED`; an owner confirmation or offline reset is still required.

The current version does not persist an acquisition journal. After an
orchestrator restart, existing acquisitions are lost. Clients must not retry,
reuse, or release buffers until the affected owners, regions, and clients are
quiesced or reset. The existing Region/Object restart rule remains authoritative:
an unclean restart requires an offline reset of the region and its clients.

## Follow-ups

- Persistent acquisition recovery and safe restart handling.
- Private-L1, Region/Object owner integration and an owner address registry.
- Partial and tail chunks; align with [#5886](https://github.com/LMCache/LMCache/pull/5886).
- Read-lease TTL and fencing beyond owner-confirmed cleanup.
- Eviction and extent reuse.
- Bundle GC and bounded retention policy.
