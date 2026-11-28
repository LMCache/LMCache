# Device codecs for LMCache MP serde

Design for [RFC #6669](https://github.com/LMCache/LMCache/issues/6669). The existing serde transforms an L1 object into a temporary buffer for an L2 adapter. To save resident CPU memory, LMCache must also be able to reserve an encoded L1 object and run a codec during GPU offload and restore. A lossless storage codec and a position-dependent GPU codec should be able to share size, identity and completion rules.

## Implemented in this change

A serde factory can construct `AsyncSerdeProcessor(serializer, deserializer, framed_variable_length=True)` if its decoder validates a complete record, including its length. The storage wrapper asks its L2 adapter for a bounded load operation at setup. The buffered filesystem adapter accepts files up to the destination capacity, checks EOF when the buffer fills, and sets the temporary object's used length before reporting a hit. The decoder sees exactly that byte range. A decoder error fails the deserialize task. Existing serdes default to exact-size reads. This opt-in requires buffered filesystem I/O in the initial implementation.

## Proposed interface

Keep the existing serde registry, serializer and deserializer entry points. Add an optional codec factory with a construction context and native batch methods. A generic MP server module uses it at the GPU transfer boundary. An adapter can also expose the codec as an existing storage serde when its execution capabilities fit.

```python
# Proposed signatures. Names are open for discussion.
build_codec(config, group_layout, device, kv_rank, local_shard) -> Codec
codec.capabilities -> Capabilities
codec.profile -> DecoderProfile
codec.max_encoded_bytes(layout) -> int
codec.workspace_bytes(layout, batch_size, direction) -> int
codec.encode_batch(sources, destinations, context) -> EncodedLengths
codec.decode_batch(sources, valid_lengths, destinations, context) -> None
```

| Responsibility | Contract |
|---|---|
| Registration | The codec declares supported layouts, execution devices per direction, fixed or variable size, position requirements, thread safety and batch limit. Validate per group at registration. A group explicitly configured as raw follows its ordinary transfer path. |
| Input | Each view describes shape, dtype, strides and group membership. The context supplies the device stream and absolute positions when requested. Skips, splits and reordered batches preserve the correspondence of positions to objects. Validate dependencies against the stored object's layer/head boundaries. |
| Allocation | The runner supplies bounded destinations. The codec returns one valid encoded length per object, which can stay on device until the next consumer needs it. Limit staging and scratch in bytes. Fixed slots work for the first L1 implementation. |
| Completion | The runner orders inputs after the producer, calls the codec and records completion on the supplied stream. A codec using auxiliary streams joins them before returning. Source, output and scratch stay live through completion. After failure, discard unpublished objects. Teardown drains in-flight work before releasing the context. |
| Batching | One codec call may process multiple objects. Each stored object must still restore independently under its declared profile. Byte determinism and invariance to batch grouping are separate capabilities. A legacy serde may use the existing single-object loop. |

## Two consumers

| Consumer | Store | Restore |
|---|---|---|
| Storage serde | Raw L1 → encode into temporary buffer → store valid bytes in L2 | Bounded L2 read → validate received length and descriptor → decode into L1 |
| GPU transfer module | Gather paged KV → encode on device → copy into compressed L1 → publish after completion | Validate L1 object → copy to device → decode → scatter into paged KV → signal readiness to inference |

For a hybrid model the transfer module can compress its attention group while preserving raw indexer bytes and separate state snapshots. For each model registration, retain the encoded layout and slot bound for each logical KV rank and object group. Use that entry to allocate the destination during L2 prefetch. One shared layout is valid only after verifying that it accommodates every participating rank and that the read path accepts their respective valid lengths.

## Memory placement on discrete GPUs and UMA

GPU KV and L1 name logical roles. On DGX Spark the engine's paged KV, LMCache's host arena and GPU scratch compete for the same physical DRAM. A retained prefix must survive engine-page recycling and worker restart. Keep its encoded representation in storage owned by LMCache and hold the worker's page lease until the last read completes. On restore, wait for the new worker's safe-write event and signal readiness after decode and final placement. [UMA accounting RFC #6206](https://github.com/LMCache/LMCache/issues/6206)

| Design choice | Initial implementation | Optional optimization |
|---|---|---|
| Buffer access | The runner uses allocator and platform information to select legal source and destination views for the executing device and process. Unknown access capabilities select the staged path. | If the codec supports the layout and the backend verifies mapped access, encode directly into an independently owned L1 slot or read from it on decode. Decide the two directions independently. |
| Admission | Bound the batch in bytes before submitting work. Include runner staging, codec workspace, persistent calibration and maximum in-flight work. On UMA charge distinct backing allocations to one shared budget and reserve space for the engine and OS. Reduce the batch or decline registration if even one object cannot fit. | Expose a platform capability in core only if existing allocator/transfer hooks cannot describe the accessible view or its lifetime. |
| Identity and accounting | Derive the decoder profile from the encoding. A choice between staged and mapped access can reuse the same key when both produce compatible records. Report occupied L1 bytes separately from committed arena backing. An imported engine mapping refers to the worker's allocation and adds a charge only for any additional backing. | Retain the faster legal path per platform after measuring store and restore separately. |

Shared DRAM alone establishes neither GPU access to a given CPU pointer nor safe reuse across processes. Direct access requires a valid mapping, addressability, layout, alignment, bounds and synchronization for the lifetime of the operation. An in-place overwrite of live engine KV would violate the ownership contract. Lossy KVTC restores normal engine KV before inference consumes it.

Capacity topology and addressability are independent inputs. For example, GB200 can expose coherent access across Grace memory and GPU HBM while each pool retains its own capacity limit. At first the operator supplies the runner's workspace allowance after reserving headroom for other owners. Cluster-wide allocation policy belongs with the engine and platform work in #6206.

## First implementation for review

| Where | Proposed interface change | First test |
|---|---|---|
| MP server plugin | An opt-in server module constructs a codec and owns the transfer loop, batching, staging and readiness event. The L2 consumer continues to use the registered serde processor. | A device identity codec round-trips a whole chunk and a legacy FP8 serde loads using its existing config. |
| Transfer hook | Expose an optional position-aware hook carrying the aligned absolute chunk offsets and the registering worker's shard identity. The default implementation delegates to the existing `BaseCacheContext.transfer_object_group` signature. | An old override still works. A recording codec sees the original offsets after a split store and partial retrieve. |
| Object identity | At registration derive a reserved codec-profile tag for each encoded group. Make it available at key construction for both lookup and store. An explicitly raw group keeps its untagged key. Profile selection must precede lookup, including when the scheduler and worker run in different processes. | Raw, profile A and equal-width profile B each miss on first lookup, then coexist and restore after restart. |
| Filesystem L2 reads | For codec-aware reads, return the actual bytes received per successful object with the load result, or provide authoritative size discovery before allocation. Maintain exact-size checks on the legacy path. | Load a valid short record into a larger destination, then separately reject a truncated record and an oversized file. |

The identity change can be reviewed and tested on CPU independently. For TP, publish the complete registration to the lookup path only when the required rank/group profiles and layouts are known. Advertise a reusable prefix only across the ranks and groups required to resume it. A profile change on one rank must yield a safe aggregate miss for incompatible data, permit writing the replacement, and allow reuse after a second restart. An ordered digest of the participating rank/group profiles is one candidate. Per-rank keys with an explicit all-required-shards check are another. The per-I/O length belongs with the read result and the record. A length change must preserve the key when decoder compatibility is unchanged.

## Cache compatibility and storage I/O

Derive a reserved matching tag from the codec, format, decoder-relevant configuration and layout at registration. Propagate it consistently through lookup, store and prefetch using [object identity tags](https://github.com/LMCache/LMCache/pull/5843). Preserve tenant isolation. An entirely raw group can keep its legacy identity. For the initial implementation, choose raw or coded groups before lookup. A later per-object fallback would need a stable lookup profile and a raw/encoded descriptor inside it.

The stored descriptor identifies the actual encoding, profile and logical length. Forward existing encoded objects across tiers as opaque bytes. Refuse incompatible secondary transforms and re-offload paths that would silently re-encode lossy data. On load, a codec-aware L2 adapter must report the actual bytes received or discover the record length before allocation. Validate framing against that length, with separate accounting for physical padding. Preserve exact-length checks for callers expecting fixed raw objects.

## Implementation and verification

Start with a vertical slice through the existing L1↔L2 storage serde. Store a framed lossless record, evict its L1 source, reopen the filesystem cache and prefetch it into a fresh L1 destination. Compare exact decoded bytes and test profile changes and corrupt records. This slice uses the same storage contract on a discrete GPU system and on UMA. Verification of L0↔L1 adds engine page ownership, device access and inference completion in the subsequent transfer-module slice.

1. Use the existing server-module and transfer hooks. Add backwards-compatible access to chunk positions and shard ownership where existing context is insufficient. Document lifecycle and unsupported modes.
2. Add focused changes for matching-tag propagation and actual-length L2 reads. With a persistent filesystem cache, change to a different decoder profile of the same width. Require a miss, a successful new store, and a subsequent hit. Valid variable-size objects must round-trip after restart. Reject truncated or oversized records.
   Include a restart with two distinct rank widths and the original calibration, then change one rank's calibration while preserving its width. Verify each rank's restored KV and actual L2 reads, and repeat after a second restart. A completed inference request can also result from recomputation.
3. Run contract tests with legacy FP8/AES-GCM, variable-size lossless compression and a calibrated GPU codec. Cover partial batches, split stores at nonzero positions, concurrent contexts, shutdown during transfer and an injected device failure. Require exact reconstruction for lossless codecs and parity with the direct implementation for lossy codecs.
4. Measure paired warm-hit overhead with the same retrieved tokens. Separately run rotating prefixes under the same CPU-cache and GPU-KV budgets to measure capacity. Report hit rate, TTFT, throughput, allocated L1, valid payload bytes and peak workspace. Include a dense TP1 model, a hybrid TP4 model and a single-device hybrid on unified memory.
5. On UMA, first compare staged raw, direct-plugin and shared-interface arms under matched settings. Record node headroom, swap/OOM activity, peak staging and actual loaded tokens. For a physical-capacity claim, compare under a measured common node allowance with explicit OS reserve, including native prefix caching as a reference. Exercise eviction and worker restart with a retained L1 object. A mapped fast path must pass the same reconstruction, completion and teardown tests as staging, then show a measured reduction in copies or workspace and an acceptable serving cost.

**Decisions for review.** The buffered path delivers length through the caller's temporary object. Do direct or remote backends also need a length result or size discovery before allocation? Should lookup use a single tag derived from the ordered rank profiles, or require compatible hits for every shard? Should the device codec factory extend serde registration or sit behind a server module? Related proposals cover [GPU decompression](https://github.com/LMCache/LMCache/issues/4498) and [hardware compression](https://github.com/LMCache/LMCache/issues/4149).
