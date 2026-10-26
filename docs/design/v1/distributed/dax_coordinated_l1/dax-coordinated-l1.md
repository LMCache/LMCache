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
For example, a configuration looks like this:

```json
{
  "region_id": "lab-shared-range-1",
  "region_epoch": 1,
  "participant_id": 0,
  "rank_placement": {
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
  "visibility_mode": "x86_clflushopt_bulk_v1",
  "skip_payload_flush": false
}
```

All participants must agree on participant count, region/epoch, layout, ownership, shared ranges,
and visibility mode. Participant IDs differ; device paths
may be host-local aliases of the same physical range. Payload flush policy,
attach memcheck, timeout and logging are host-local settings.

### Required settings

| Setting | Meaning |
| --- | --- |
| `region_id` | Operator-assigned name for the shared arena; all participants must match the stored identity. |
| `region_epoch` | Positive fencing epoch stored in the shared arena; all participants must match. Changing it does not reset existing metadata. |
| `participant_id` | Unique MP server ID from `0` to `participant_count - 1`; participant 0 initializes unformatted metadata, while peers wait and attach. |
| `rank_placement` | For 1..8 KV ranks; all device paths, metadata reservations and payload ranges live in `regions`; see [KV rank placement](#kv-rank-placement) |

### Optional settings

| Setting | Default / meaning |
| --- | --- |
| `participant_count` | `2`; sets the maximum number of MP participants sharing the arena. Supports `2` or `4`, with one unique participant ID per MP server. |
| `ownership_mode` | `equal`; alternatively `participant_0_all` |
| `buckets_per_level` | `[200003, 200009, 200017, 200023, 200029]`; one to eight positive level sizes |
| `visibility_mode` | `x86_clflushopt_bulk_v1` (default): flush/invalidate payload cache lines with CLFLUSHOPT and one SFENCE per batch. `x86_clflush_64b_v1`: use CLFLUSH and MFENCE per payload range. Metadata synchronization is unchanged. With payload maintenance enabled, bulk mode reduces fence overhead and is generally faster for large batches; `skip_payload_flush=true` skips this work in both modes. |
| `skip_payload_flush` | `false` (default): flush payload after GPU-to-DAX DMA and refresh it before DAX-to-GPU DMA. `true`: skip both operations, reducing CPU cache-maintenance overhead and generally improving effective payload transfer performance; DMA bytes and completion synchronization are unchanged. Recommended `true` to fully utilize the media’s transfer performance once all prerequisites are satisfied, including qualified DDIO non-allocating writes, UC coverage of the entire payload on every host.
| `reformat` | `false` (default). Set `"reformat": true` in this JSON on participant 0 only to erase existing cache metadata or layout. Ensure that Stop all participants first, then start participant 0 with `"reformat": true` and wait for initialization before starting peers. Active peers are not detected, so user ensure all participants are terminated. |
| `memcheck_on_attach` | `false` (default). Set to `true` to check metadata integrity before reuse after an abnormal shutdown or when corruption is suspected. Scans the full index while all participants have paused cache operations; does not repair corruption. Success: no dedicated check-result message; attach continues and logs `Device-DAX model payload geometry: ...`. Failure: attach is rejected with `RuntimeError: DAX-Coordinated L1 region requires recovery before attach`; no bucket/slot details are reported. |
| `per_transfer_logging` | `false`; completed STORE/RETRIEVE bytes, latency and GB/s |
| `read_view_cache_max_entries` | `8192`; host-local descriptor cache limit. `0` disables retention without changing payload capacity. |
| `hardware_qualification_digest` | Optional identifier for an operator-managed hardware qualification record; used to check that a participant references the same record as the shared arena. Does not validate hardware settings. |

## Layout registration process

```text
Server startup
  -> DaxCoordinatedL1Client -> validate DAX region configuration
REGISTER_KV_CACHE
  -> LMCacheDrivenTransferModule.register_kv_cache()
  -> if storage_manager.uses_dax_coordinated_l1
  -> storage_manager.dax_coordinated_l1_backend.client.initialize_model_layouts()
  -> resolve engine KV world size, rank slices and model geometry
  -> payload mmap + GPU registration -> metadata mmap -> format or attach
  -> stores / reads
```

Participant 0 initializes metadata if it is not already formatted, or explicitly
reformats it when `reformat=true`.
Other participants retry attachment on subsequent requests until
initialization completes.

New participants must register the same model identifier, KV world size,
chunk size, and tensor layouts as the initializer. Attach also validates
the metadata format, participant count, region identity, slot geometry,
ownership, and mapping bounds. Matching values preserve the existing
state; incompatible metadata is rejected without modification.
Changes that invalidate the existing layout require all participants
to stop and the region to be explicitly reformatted.

Storage size is calculated from the tensor shapes and dtypes registered by
the engine. Compressed KV and sliding-window state are stored according to
their actual tensor sizes.
The configured payload regions (payload_offset_bytes, payload_size_GiB) provide the storage space.
Each region is partitioned into its assigned rank slices; the total number of slices across
all regions equals the engine-reported KV world size.
Each slice contains a separate pool for every registered object group.
Slots have a fixed size within each pool. Group pools in the same
rank slice have equal slot counts and independent free lists; unused
capacity cannot be borrowed across pools.

An object group (ObjectGroupInfo, defined by KVLayerGroupsManager) is a set of
KV tensors registered as one storage and retrieval unit.

For Qwen3 with full attention, the K/V tensors form one object group per KV
rank. The following example uses one payload region and engine-reported KV
world size 2 (vLLM TP2, PP1, DCP1). Each rank stores its own KV-head shard;
each slot holds one chunk's K and V tensors across the registered layers.

```text
Payload region (payload_offset_bytes, payload_size_GiB)
+------------------------------------------------------------------+
| KV rank slice 0                                                  |
|                                                                  |
|  Object group 0 pool: K + V for KV rank 0                         |
|  [ slot 0 ][ slot 1 ][ slot 2 ] ... [ slot N-1 ]                   |
|                                                                  |
+------------------------------------------------------------------+
| KV rank slice 1                                                  |
|                                                                  |
|  Object group 0 pool: K + V for KV rank 1                         |
|  [ slot 0 ][ slot 1 ][ slot 2 ] ... [ slot N-1 ]                   |
|                                                                  |
+------------------------------------------------------------------+
KV world size = 2; each GPU accesses its own KV rank slice.
```

For DSV4 Flash-0731 with `separate_object_groups=true`, group 0 contains
compressed KV and indexer data needed across the cached history; group 1 contains
sliding-window KV and compressor state needed for the recent window.

The following example has one payload region and engine-reported KV world size
1 (vLLM TP2, PP1, DCP1). Both GPUs read the same stored KV shard:

```text
Payload region (payload_offset_bytes, payload_size_GiB)
+------------------------------------------------------------------+
| KV rank slice 0 (KV world size = 1)                               |
|                                                                  |
|  Object group 0 pool: compressed KV + indexer data                |
|  Full-history data                                               |
|  [ slot 0 ][ slot 1 ][ slot 2 ] ... [ slot N-1 ]                   |
|                                                                  |
|  Object group 1 pool: sliding-window KV + compressor state         |
|  Recent-window data                                              |
|  [       slot 0       ][       slot 1       ] ... [ slot N-1 ]     |
|                                                                  |
|  Same slot count N; slot sizes differ between groups.             |
+------------------------------------------------------------------+
```

With `separate_object_groups=false`, these tensors form one object group and
use one pool within the rank slice.

## KV rank placement

Users who know the physical topology of their memory pool can place
payload regions on selected devices or address ranges. KV ranks are
assigned to these regions in round-robin order, distributing rank-specific
accesses across the configured regions and allowing independent transfers
to proceed in parallel.

KV world size comes from the engine; configure only physical regions.
Rank `r` uses `regions[r % len(regions)]`; ranks sharing a region receive
equal, disjoint whole-GiB slices. Only `regions[0]` declares metadata.

### Example: two payload regions

For Qwen3 with PP=1, DCP=1 and one MP server per host, this JSON supports
TP2 or TP4 (KV world size 2 or 4). The two configurations cannot share the
same initialized arena; switching between them requires reformatting.

```json
{
  "region_id": "lab-shared-range-1",
  "region_epoch": 1,
  "participant_id": 0,
  "rank_placement": {
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
  },
  "skip_payload_flush": true
}
```

- `rank_placement`: required; supports KV world size 1..8 with PP=1 and all
  ranks local to one MP server per host. All ranks and groups share one index.
- `regions`: 1..KV world size entries. Hosts must identify the same physical
  ranges; local device paths may differ.
- `metadata_offset_bytes` / `metadata_reservation_bytes`: here, reserve 1 GiB
  starting at offset 0. Only the first region declares these fields; omit them
  or use `null` in later regions. The reservation must be positive, page-aligned,
  large enough for runtime metadata, and disjoint from payload.
- `payload_offset_bytes` / `payload_size_GiB`: here, allocate 64 GiB per region
  starting at 30 GiB and 256 GiB, independently of `--l1-size-gb`.
  [Ownership](#ownership) applies within each rank slice.

Both regions are on `/dev/dax0.0`. For TP1, remove the second region from
the JSON, leaving 64 GiB total. TP2 and TP4 use both regions (128 GiB total).
The following allocation is before participant ownership is applied:

| Qwen3 configuration | Region 0: [30, 94) GiB | Region 1: [256, 320) GiB |
| --- | --- | --- |
| TP1 / KV world size 1 | rank 0: 64 GiB | Remove from JSON |
| TP2 / KV world size 2 | rank 0: 64 GiB | rank 1: 64 GiB |
| TP4 / KV world size 4 | rank 0: [30, 62), rank 2: [62, 94); 32 GiB each | rank 1: [256, 288), rank 3: [288, 320); 32 GiB each |

### Example: DSV4 TP2 with one KV rank

Use two hosts, each with two GPUs and one MP server, running the same model
revision, runtime and LMCache source. TP=2, PP=1 and DCP=1 give KV world size 1.

Save the following as `dax.json` on host A, using ranges reserved for your
deployment. `skip_payload_flush=true` requires the
[no-flush prerequisites](#cross-host-kv-cache-coherency).

```json
{
  "region_id": "dsv4-kv1-example",
  "region_epoch": 1,
  "participant_id": 0,
  "participant_count": 2,
  "rank_placement": {
    "regions": [
      {
        "devdax_path": "/dev/dax0.0",
        "metadata_offset_bytes": 5368709120,
        "metadata_reservation_bytes": 536870912,
        "payload_offset_bytes": 118111600640,
        "payload_size_GiB": 8
      }
    ]
  },
  "skip_payload_flush": true
}
```

On host B, copy the same JSON and change only `participant_id` to `1` (the
local device path may differ if it identifies the same shared range).

Start one MP server per host in its own terminal, then start A's vLLM engine
before B's so participant 0 can register the layout and format fresh metadata.
Cross-host cache sharing uses shared DAX metadata and payload, not these network
ports.

```bash
python -m lmcache.cli.main server \
  --host 127.0.0.1 --port 18820 \
  --http-host 127.0.0.1 --http-port 19820 \
  --chunk-size 256 --l1-size-gb 8 --no-l1-use-lazy \
  --shm-name '' --eviction-policy noop \
  --l1-devdax-path /dev/dax0.0 \
  --separate-object-groups \
  --dax-coordinated-l1-config-json "$(cat dax.json)"
```

In a second terminal on each host, set `DSV4_MODEL_DIR` to the local checkpoint
directory. Use the same model path inside both runtimes: the MP connector uses
the model name/path for cache identity. `--served-model-name` sets the API model
name and does not change cache identity.

```bash
export DSV4_MODEL_DIR=/models/DeepSeek-V4-Flash-0731
vllm serve "$DSV4_MODEL_DIR" \
  --served-model-name deepseek-ai/DeepSeek-V4-Flash-0731 \
  --host 127.0.0.1 --port 19420 \
  --tensor-parallel-size 2 --enable-expert-parallel --disable-custom-all-reduce \
  --kv-cache-dtype fp8_ds_mla \
  --trust-remote-code --block-size 256 \
  --max-model-len 8192 --max-num-seqs 1 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager --seed 42 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":18820,"lmcache.mp.mq_timeout":1800,"lmcache.mp.mp_transfer_mode":"lmcache_driven"}}'
```

For remote reuse, send identical prompt token IDs and `cache_salt` to A and B,
waiting for A's STORE completion before B's request. With 1,056 input tokens,
expect 1,024 external-cache hit tokens on B and zero GPU prefix-cache hits.
Output-token equivalence has not fully passed validation; mismatches remain
unresolved, so cache hits alone do not establish inference correctness.

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
participant 0 all allocation rights. This policy is applied within each KV rank slice. In both modes, all participants can read objects in peer-owned slots under the
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

- Each arena supports one model/layout profile with an ordered list of object groups.
- Now up to 4 MP participants are supported.
  KV world size must be 1..8 with PP=1.
- Allocation is limited to each participant's defined slots.
  Free capacity belonging to another participant cannot be borrowed
  when that slice is full.
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
layout/lifecycle/contention, KV rank routing and mocked device-range/CUDA-registration
failures. They are in `tests/v1/distributed/test_dax_coordinated_l1.py` and
`tests/v1/distributed/test_dax_coordinated_l1_native.py`.
Device identity and CUDA registration are mocked; native tests use shared
memory mappings. These tests do not perform a hardware round trip.
Cross-host visibility, payload correctness and multi-GPU performance require
separate hardware qualification for the actual topology and configuration.
