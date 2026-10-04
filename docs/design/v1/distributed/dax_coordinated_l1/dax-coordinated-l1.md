# DAX-Coordinated L1

`DaxCoordinatedL1Backend` lets multiple MP servers share KV objects through the
same physical Device-DAX range. The shared index, slot ownership records and
cross-host synchronization state reside. Participants coordinate
allocation and object lifetime through this metadata without per-operation
metadata RPCs; GPU payload transfers use direct DMA.

```text
+--------------------+              +----------------------------+              +--------------------+
| MP participant 0   |              | Shared metadata (DAX)      |              | MP participant 1   |
|                    |              |                            |              |                    |
| L1Manager          | coordination | Key -> slot mapping        | coordination | L1Manager          |
| DAX-Coordinated L1 |<------------>| Slot-wise metadata         |<------------>| DAX-Coordinated L1 |
|         ^          |              +----------------------------+              |         ^          |
|         |          |                             |                            |         |          |
|         |          |                             | slot mapping               |         |          |
| Transfer control   |                             v                            | Transfer control   |
|         |          |              +----------------------------+              |         |          |
|         |          |              | Shared KV objects (DAX)    |              |         |          |
|         v          |              | KV chunk A                 |              |         v          |
|       GPU(s)       |<----DMA----->| KV chunk B                 |<----DMA----->|      GPU(s)        |
|                    |              | KV chunk C  ...            |              |                    |
+--------------------+              +----------------------------+              +--------------------+

  DMA: Direct KV data transfer between GPU memory and DAX.
  Transfer control: MP submits transfers and handles completion callbacks.
  coordination: CPU load/store and cache maintenance on shared metadata.
```

both MPs access these common KV objects and the same metadata. Ownership governs allocation and deletion; either MP can read any published object. Each MP schedules its local GPU transfers and handles DMA completion. Metadata and payload mappings may reside on DAX device.

## Activation and configuration

Build with `BUILD_WITH_DAX_COORDINATED_L1=1` and supply
`--dax-coordinated-l1-config-json`. Without that JSON, a Device-DAX path
selects the private allocator. Without the native extension, enabling the
DAX-Coordinated L1 backend fails with a rebuild instruction.

`DaxCoordinatedL1Config` lives in `lmcache/v1/distributed/config.py` and parses
JSON through `from_json()`. The local `--l1-devdax-path` must match
`rank_placement.regions[0].devdax_path`.
For TP=1, a configuration looks like this; replace the qualification placeholder
with the SHA-256 digest of the actual hardware qualification record:

```json
{
  "region_id": "lab-shared-range-1",
  "region_epoch": 1,
  "participant_id": 0,
  "hardware_qualification_digest": "<64 hex characters>",
  "rank_placement": {
    "tp_size": 1,
    "regions": [
      {
        "devdax_path": "/dev/dax0.0",
        "metadata_offset_bytes": 0,
        "metadata_reservation_bytes": 1073741824,
        "payload_offset_bytes": 32212254720,
        "payload_size_GiB": 64
      }
    ]
  },
  "ownership_mode": "equal",
  "visibility_mode": "x86_clflush_64b_v1",
  "skip_payload_flush": false
}
```

All participants must agree on participant count, region/epoch, layout, ownership, shared ranges,
visibility mode and qualification digest. Participant IDs differ; device paths
may be host-local aliases of the same physical range. Payload flush policy,
attach memcheck, timeout and logging are host-local settings.

| Setting | Default / meaning |
| --- | --- |
| `region_id` | Required operator-assigned name for the shared arena; all participants must match the stored identity. |
| `hardware_qualification_digest` | Required SHA-256 hex digest of the hardware validation record, identical across participants; does not automatically verify DDIO or UC. |
| `participant_count` | `2`; supports `2` or `4`, one unique ID per MP server |
| `participant_id` | Required unique MP server ID from `0` to `participant_count - 1`; participant 0 initializes unformatted metadata, while peers wait and attach. |
| `ownership_mode` | `equal`; alternatively `participant_0_all` |
| `buckets_per_level` | `[200003, 200009, 200017, 200023, 200029]`; one to eight positive level sizes |
| `visibility_mode` | `x86_clflush_64b_v1`; alternatively `x86_clflushopt_bulk_v1` |
| `skip_payload_flush` | `false`; qualified opt-in to skip STORE and RETRIEVE payload cache maintenance |
| `memcheck_on_attach` | `false`; opt-in full-index diagnostic scan with all participants quiesced |
| `per_transfer_logging` | `false`; completed STORE/RETRIEVE bytes, latency and GB/s |
| `rank_placement` | Required for TP=1..8; all device paths, metadata reservations and payload ranges live in `regions`; see [TP support](#tp-support) |

- Llama or Qwen3 model names, one homogeneous object group and one runtime
  `[K/V, layers, tokens, hidden]` layout. Qwen3.5 and other model names
  outside the allowlist, as well as multiple object groups, are rejected.
- Explicit `rank_placement.tp_size` supports local TP=1..8 with one
  shared index and a designated payload slice per rank.

## Model registration and layout

```text
Server startup
  -> DaxCoordinatedL1Client -> payload mmap + GPU registration
REGISTER_KV_CACHE
  -> LMCacheDrivenTransferModule.register_kv_cache()
  -> if storage_manager.uses_dax_coordinated_l1
  -> storage_manager.dax_coordinated_l1_backend.client.initialize_model_layouts()
  -> resolve model profile and slot geometry -> metadata mmap -> format or attach
  -> stores / reads
```

The registered `MemoryLayoutDesc` shape and dtype determine object bytes;
there is no model-size table. The exact model name, KV world size, chunk size
and runtime layout bind the profile. Identical re-registration is idempotent;
a different profile is rejected even if it has the same byte size.

```text
slot_bytes = align_up(object_bytes, 64)
rank_slot_count = floor(rank_payload_bytes / slot_bytes)
equal:             round each rank's count down to a multiple of participant_count; divide equally
participant_0_all: use all complete slots; nonzero participants own zero
slot_count = sum(rank_slot_counts)  # TP=1 has one payload slice
```

DAX-Coordinated L1 validates the participant count, region identity, slot geometry
and layout compatibility before attaching to existing metadata. Incompatible
metadata is rejected without modification. Changes that invalidate the existing
layout require all participants to stop and the region to be explicitly
reformatted. No live upgrade or automatic recovery is provided.
The payload is mapped separately. The superblock binds region/epoch, geometry,
layout and qualification digests; bucket and slot generation links validate
object identity in both directions.

Participant 0 formats metadata without the format magic and publishes that magic
last, after the initialized metadata is visible. Nonzero participants retry attach
until formatting completes. Payload mappings and GPU registrations remain alive
across these retries; closing before attach releases them as well. Attachment
is serialized with model registration and shutdown within each MP,
so a concurrent retry cannot replace a core that already holds reservations.
At every TP size, the model-dependent metadata extent must fit the first
region's reservation before metadata mmap. Existing formatted metadata must match the expected
profile; incompatible regions are rejected rather than overwritten.

Attach always validates the superblock magic/version, region/epoch and layout
contract, participant ownership and mapping bounds. `memcheck_on_attach=true`
additionally scans all bucket states and bucket/slot links; use it only while
all participants are quiesced, since concurrent updates can appear inconsistent.
The default skips this scan to allow attachment during normal traffic. Explicit
`memcheck()` calls and per-operation state/generation checks remain available.

## TP support

TP=1 and TP>1 use the same client and lifecycle methods, with one native index
and one logical slot namespace. Configure rank placement inside the JSON passed
to `--dax-coordinated-l1-config-json`:

| JSON field | Requirement and behavior |
| --- | --- |
| `rank_placement` | Required for local TP=1..8 with PP=1. |
| `rank_placement.tp_size` | Integer from 1 to 8, checked against the KV world size at registration. Set the engine's tensor parallelism separately. |
| `rank_placement.regions` | Ordered device paths and payload ranges; 1 to `tp_size` entries. TP=1 has exactly one region. Regions may use disjoint ranges of one device or separate devices. |
| `regions[0].metadata_offset_bytes` | Required metadata start, including an explicit `0` for the device beginning. |
| `regions[0].metadata_reservation_bytes` | Required positive, page-aligned reservation size; 1 GiB in these examples. |
| `regions[].payload_offset_bytes` / `payload_size_GiB` | Required payload start and capacity per region, independent of `--l1-size-gb`. |

The first region alone supplies both metadata fields. Later regions must omit
both fields (or use `null`); even a second metadata offset of `0` is rejected.
This rule also applies when multiple regions name the same device.

`rank % len(regions)` selects a region; ranks sharing a region receive disjoint,
equal whole-GiB payload slices. For two 256 GiB regions A and B:

| TP | Rank placement |
| --- | --- |
| 2 | rank 0 → A (256 GiB); rank 1 → B (256 GiB) |
| 4 | rank 0/2 → A (128 GiB each); rank 1/3 → B (128 GiB each) |

As in the existing MP key scheme, the same token chunk has the same
`chunk_hash` across ranks; `ObjectKey.kv_rank` distinguishes each rank's KV
object. The shared index hashes the complete key, including `kv_rank`.

The client validates the rank encoded by `ObjectKey.ComputeKVRank`. New
allocations use that rank's free list; reads and deletion follow the
global slot ID in the shared index. An address table maps slots to local payload
addresses. Each object stays contiguous; a full slice does not borrow space
from another rank.

The common index starts at `regions[0].metadata_offset_bytes` and reserves
`regions[0].metadata_reservation_bytes` once, independent of TP size.
The homogeneous KV layout determines one common slot size; bucket counts
cover all ranks together. To assign each rank an explicit offset on one device,
provide exactly `tp_size` regions: `regions[i]` then belongs to rank `i`.
This TP=2 example uses two disjoint ranges of `/dev/dax0.0` at offsets
[30, 94) and [256, 320) GiB. Its no-flush setting assumes the deployment has qualified DDIO
non-allocating writes and UC coverage for every payload range on all hosts:

```json
{
  "region_id": "lab-shared-range-1",
  "region_epoch": 1,
  "participant_id": 0,
  "hardware_qualification_digest": "<64 hex characters>",
  "skip_payload_flush": true,
  "rank_placement": {
    "tp_size": 2,
    "regions": [
      {
        "devdax_path": "/dev/dax0.0",
        "metadata_offset_bytes": 0,
        "metadata_reservation_bytes": 1073741824,
        "payload_offset_bytes": 32212254720,
        "payload_size_GiB": 64
      },
      {
        "devdax_path": "/dev/dax0.0",
        "payload_offset_bytes": 274877906944,
        "payload_size_GiB": 64
      }
    ]
  }
}
```

| Rank | Device | Payload offsets (GiB) |
| --- | --- | --- |
| 0 | `/dev/dax0.0` | [30, 94) |
| 1 | `/dev/dax0.0` | [256, 320) |

Each rank receives 64 GiB (128 GiB total) before participant ownership is applied. Only the
first entry declares metadata; repeating the device path does not create
additional metadata arenas. Replace the qualification placeholder for the
deployment.

The metadata reservation is `[metadata_offset_bytes,
metadata_offset_bytes + metadata_reservation_bytes)` on the first device.
No payload may overlap it. Only the device-aligned, model-dependent metadata
size is mapped and initialized, not the entire reservation.

For existing JSON, move the top-level `devdax_path`, `metadata_offset_bytes`,
`payload_offset_bytes` and `payload_size_GiB` into the region configuration.
Move `rank_placement.metadata_reservation_bytes` into `regions[0]` as well.
Old locations are rejected rather than silently ignored. An unchanged TP>1
placement retains its v1 layout digest. Former TP=1 arenas require fresh ranges
or an offline reset because TP=1 now includes the placement digest too.

Each worker transfers through its existing MP GPU and CUDA stream to its slice;
CUDA device ordinals need not equal TP ranks. Portable CUDA registration and the
STORE/RETRIEVE completion ordering below apply to every rank. Shutdown drains all
visible CUDA devices before releasing reservations and mappings.

The common [host ownership policy](#ownership) applies
within each rank slice. All TP ranks must be local to one LMCache server per
host, with PP=1. Both hosts must agree on TP size, model layout and physical
placement; local device paths may differ. The persistent layout digest rejects
incompatible configurations. Reconfiguration requires fresh reserved ranges or
an operator-managed reset.

## Cross-Host KV Cache Coherency

**For this backend, enabling `skip_payload_flush=true` requires both
DDIO non-allocating I/O writes and UC for the payload regions on
every participating host.** Configure both externally before enabling the
option; the runtime neither applies nor verifies these settings.

With `skip_payload_flush=false` (the default), payload flush/refresh
supports operation without this no-flush configuration, but its additional
cost can reduce effective DMA transfer throughput.

| Policy | Payload maintenance |
| --- | --- |
| Default | Flush after GPU-to-CXL DMA completes; refresh before CXL-to-GPU DMA begins |
| `skip_payload_flush=true` | Skip both operations; retain metadata synchronization |

Payload maintenance uses `CLFLUSH -> MFENCE` or bulk `CLFLUSHOPT -> SFENCE`.
Before skipping it, clear old payload cache lines and establish a DMA-only
payload path with cross-host visibility at DMA completion.

### Why UC accompanies the no-flush policy

DDIO non-allocation controls I/O write allocation; effective UC prevents CPU
cache fills. Inbound reads can still use local caches ([Intel DDIO][intel-ddio]).
Without cross-host invalidation, a reader host's cached copy can therefore
supply stale bytes to its GPU after a peer's completed write. A writer-side
flush need not invalidate that reader-side copy.

These settings do not replace DMA completion or READY publication ordering.
Keep the default policy until the actual
deployment passes GPU byte comparisons with payload maintenance disabled.

[intel-ddio]: https://www.intel.com/content/www/us/en/developer/articles/technical/ddio-analysis-performance-monitoring.html

## KV Cache Lifecycle and Cross-Host Synchronization

Our CXL slot metadata use these states similar to original L1 states during RETRIEVE and STORE(new).

| State | Meaning |
| --- | --- |
| `FREE` | Available for new allocation |
| `READY` | Published object; readable through an `OPEN` gate |
| `WRITE_LOCKED` | Reserved for an exclusive operation; gate `CLOSED` |
| `RECOVERY` | Normal access blocked pending recovery |

DAX-Coordinated L1 tracks per participant reader gate flag separately, so no `READ_LOCKED` state is needed. (`READY` with a reader gate flag set indicates outstanding read reservations.) Readers publish activity and revalidate the state before DMA in order that writer modified the slot;
deletion checks reader activity before taking the writer lock and rejects the
deletion request if any reader is active. It rechecks after closing the gate;
if a reader entered in between, it reopens the gate, releases the lock and
rejects deletion.
Published keys are immutable: further writes are rejected. Replacing an object
requires owner deletion followed by a new allocation; there is no in-place update.


### Reader and writer stages

| Step | Reader (RETRIEVE) | Writer (STORE) |
| --- | --- | --- |
| 1 | Lookup key and find a `READY` candidate | Compute candidate bucket & precheck `FREE` |
| 2 | Publish reader activity | Acquire Peterson tournament lock |
| 3 | Recheck `READY`, same key, generations and slot ID | Recheck `FREE` and publish `WRITE_LOCKED` |
| 4 | H2D: DAX → GPU | D2H: GPU → DAX |
| 5 | H2D done → release reader | D2H done → publish payload → publish `READY` → unlock |

Reader steps 1–3 run inside ZMQ `LOOKUP`; step 4 runs during `RETRIEVE`.
Writer steps assume the internal key lookup reported the key absent.

Reader activity and the writer lock are held from step 2 through step 5.
Normal readers use activity tracking, not the tournament lock. Failed reader
revalidation withdraws activity before DMA; it checks `READY/OPEN`, key,
bucket/slot generations, slot ID and forward/back mapping.

Writer step 1 refreshes identity (64 B) and control (64 B). Non-`FREE` with the
same key returns `WriterBusy` without locking; another key or a failed lock
attempt moves to the next level. Step 3 rechecks `FREE` under the lock, reserves
an owner-local payload slot, and publishes the write intent and slot mapping.

### Peterson tournament lock

Peterson's lock is a software mutual-exclusion algorithm for two participants.
Each publishes its intent to enter through a shared flag; a shared `victim`
variable determines who yields when both contend, allowing only one to enter
the protected section at a time. This implementation backs off on contention
instead of waiting.

To support multiple MPs, these two-participant locks are arranged in a
tournament tree, with one Peterson lock at each internal node.
Each participant is an MP process.
Each bucket admits one local thread per participant. Four participants use
this tree; two use only node 0:

```text
                  node 0 (root)
                  /           \
             node 1           node 2
             /    \           /    \
            P0    P1         P2    P3

Acquire: leaf -> root       Release: root -> leaf
```

Each node publishes `flag[side]=1`, then `victim=side`. After refreshing peer
state, `flag[other] && victim==side` means failure: clear the local flag and
release acquired nodes in reverse order. Root-first release prevents a sibling
from overwriting a still-owned root flag.

Metadata refresh/publication uses `CLFLUSH → MFENCE`; 64-byte alignment does
not make separate loads atomic. If lookup sees an invalid state combination,
it briefly tries the bucket lock to recheck: contention means retry; a matching
key with a still-invalid state requires recovery. This diagnostic lock is
released before normal reservation.

### Ownership

Ownership in this backend defines where an MP participant may reclaim KV Cache payload slots for incoming write(new) operation; they do not allocate KV objects in advance. A new object consumes a free slot when its write reservation succeeds.

Participants are MP processes, each with a unique participant ID. `equal`
divides allocation rights among all participants; `participant_0_all` gives
participant 0 all allocation rights. With TP enabled, this policy is applied within each rank slice. In both modes, all participants can read objects in peer-owned slots under the
shared reader protocol.

For example, with `participant_count=4` and `ownership_mode="equal"`, 128 GiB of usable slot capacity gives each MP participant a 32 GiB allocation budget. This division stores no KV data by itself. With `participant_0_all`, nonzero participants can read
existing objects, but cannot allocate or reclaim slots. It is just for experiment.

### Deletion

delete_key() attempts to delete an object in two steps:

1. **Reader precheck:** Check for active readers Reject deletion if any participant has an active reader.
2. **Deletion:** Acquire the writer lock, revalidate the object and mapping,
   then close the gate and recheck readers. If any reader is active, reopen
   the gate and release the lock without deleting; otherwise reclaim the slot
   and release the lock.

Only the slot owner may delete an object. Now automatic DAX L1 eviction is unsupported; a full owner shard returns `OUT_OF_MEMORY`.

## Status, limits and validation

Status and usage are reported per participant, based on its owned slots and
local reservations. Shared read and deletion safety is enforced separately
by the native metadata protocol.

Capacity reports retain the configured owner budget. After model registration,
runtime allocation is limited to the usable capacity of complete slots.

### Current limitations

- Hardware testing exercised Qwen3-8B with FP8 KV at TP=1. Other Llama/Qwen3
  models meeting the [same layout constraints](#activation-and-configuration) are
  expected to work, but were not hardware-validated in these experiments.
  Each arena supports one model profile and homogeneous KV object group;
  changing the registered model or layout requires a fresh or explicitly
  reformatted arena.
- Up to four MP participants, with `participant_count` configured as either 2 or 4.
  TP=1 or configured local TP=2..8 with PP=1.
  All TP ranks on a host must use one LMCache server.
- Allocation is limited to each participant's designated slots for the target
  TP rank. Free capacity belonging to another participant or rank cannot be
  borrowed when that slice is full.
- Cache reuse covers complete chunks only. A trailing partial chunk must be
  computed by the serving engine.
- L2 adapters, hybrid DRAM overflow, GDS L1, lazy allocation and POSIX SHM
  cannot be combined with this backend. Runtime L2 attachment is also rejected.
- Automatic eviction, force deletion and global `clear()` are unsupported.
  Reclaiming committed objects requires explicit deletion by the owner participant.

### Follow-up TODOs

DAX should keep its existing per-MP allocation limits while reporting
STORE/DELETE events with `shared=True`. Each MP should also report the same
whole-pool capacity with `shared=True`, so the coordinator counts the pool and
each shared object once. A future shared L1 eviction controller would use
STORE/DELETE events to track objects and ACCESS events to track recency across
MPs when selecting eviction candidates. The current coordinator eviction
controller handles L2 only; shared DAX L1 eviction remains a separate extension.
This PR defers those reporting changes to minimize common-code modifications: it retains per-MP owner capacity declarations and L1 object events with `shared=False`. A follow-up PR would be updating the common capacity and event reporting paths together, with capacity and visibility.

Repository tests cover opt-in builds, configuration/model geometry, native
layout/lifecycle/contention, TP routing and mocked device-range/CUDA-registration
failures. They are in `tests/v1/distributed/test_dax_coordinated_l1.py` and
`tests/v1/distributed/test_dax_coordinated_l1_native.py`.
Device identity and CUDA registration are mocked; native tests use shared
memory mappings. These tests do not perform a hardware round trip.
Cross-host visibility, payload correctness and multi-GPU performance require
separate hardware qualification for the actual topology and configuration.
