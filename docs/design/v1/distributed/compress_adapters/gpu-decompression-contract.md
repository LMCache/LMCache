# Portable Record GPU Decompression Contract

## Status

This document defines the target GPU decompression contract and the seven-PR delivery plan below. The portable record format, Python reference codec, device adapters, output leases, requests, and capability profiles exist under `lmcache/v1/distributed/compress_adapters/`. Backend submission and completion handling are not implemented, and no production serde, L2, or retrieve path uses these components yet. PR1 defines the [portable record format](portable-record-format.md); this PR adds only the shared contracts assigned to PR2. Later lifecycle and integration sections describe target behavior for subsequent PRs.

The undeployed draft wire format now carries a checksum over compressed payload bytes and enforces GPU-compatible payload alignment. A standalone compatibility probe submits the frozen records to either vendor library. nvCOMP 5.3.0.16 successfully decompressed both raw-Deflate and Gzip fixtures on an RTX 4060 with CUDA 12.9.86. The remaining portability prerequisite is to run the same probe against hipCOMP 2.3 on a real AMD GPU host. These small single-chunk fixtures establish framing and no-repacking compatibility, not production-scale behavior.

If compatibility with any existing draft record becomes necessary, the wire change requires a new record version. Otherwise, version 1 may be revised while it remains undeployed, with all frozen vectors updated together.

## Purpose

LMCache should store a backend-independent compressed record, copy its exact bytes to an accelerator, and expand it into the existing contiguous KV staging buffer:

```text
portable record in L1
  -> backend-owned exact H2D copy
  -> GpuDecompressBackend
  -> validated contiguous device KV staging
  -> existing paged-KV placement
```

The production feature scope includes both `NvcompBackend` and `HipcompBackend` behind the same shared interfaces. Their implementations may land incrementally, but hipCOMP is an explicit RFC deliverable rather than unspecified follow-up work, and generic code cannot acquire CUDA- or HIP-specific assumptions.

The portable record is the interoperability boundary: it identifies Deflate and its framing, not the library that encoded or decodes it. The backend is a runtime choice and is never written into the record.

## Goals

- Keep CUDA, HIP, nvCOMP, and hipCOMP types inside vendor implementations.
- Reject unsupported record/backend combinations before native submission.
- Submit independently compressed record chunks as a device batch.
- Verify compressed-payload integrity before native decompression.
- Verify every output chunk's status, exact size, and CRC before KV is usable.
- Preserve the existing contiguous staging and paged-placement path.
- Make buffer ownership, finalization, and asynchronous completion explicit.

## Non-goals

- Selecting the production store-side compressor, framing, or chunk size.
- Connecting the portable format to serde or L2 in the interface-only slice.
- Loading directly from L2 into device memory.
- Decompressing directly into non-contiguous paged KV blocks.
- Silently falling back to CPU decompression.
- Treating CRC-32 as authentication for attacker-controlled records.

## Integrity and trust boundary

Native Deflate/Gzip decompression cannot be treated as a memory-safe validator. nvCOMP documents that corrupt input can cause undefined behavior and does not guarantee a useful per-chunk error status. Therefore, arbitrary bytes from an untrusted writer must not reach a native GPU decompressor.

The draft record carries a `compressed_payload_crc32` protected by the header checksum. It covers every byte in `[header_size, record_size)`, including deterministic zero alignment padding. The deferred L1 path verifies this checksum on the host before H2D copy or native submission.

Full-record validation returns an immutable `ValidatedCompressedRecord` containing the parsed header, immutable record bytes, and verified payload checksum. Public validation requires immutable `bytes` and retains them without copying; mutable or otherwise aliased input is rejected before parsing. This avoids an unbounded defensive copy and gives the validated value a stable owner that callers cannot release. A decompression item accepts this value rather than a bare header. Header-only parsing remains available for size discovery, but its result is not eligible for native submission.

The later zero-copy L1 path uses a guarded validated-record form that retains the source `MemoryObj` read lock. A read-only memoryview alone is insufficient because another alias could mutate its underlying owner after validation.

This checksum detects accidental storage or transport corruption. It does not authenticate malicious input because an attacker can recompute it. A deployment that permits untrusted writes must first authenticate the complete record or pass it through a memory-safe bitstream-validation boundary. Otherwise GPU decompression is disabled for that storage path.

After decompression, every backend computes CRC-32/IEEE over each output chunk and compares it with the descriptor's `uncompressed_crc32`. Native status and actual output size alone do not establish content integrity.

Each descriptor's `compressed_size` covers exactly one complete codec stream; producers do not append bytes after that stream's end marker. Structural parsing and the compressed-payload CRC cannot prove codec-stream termination. The reference decoder enforces the rule, while a native backend rejects trailing input when its library reports consumed length or trailing-data status. A backend whose native API cannot report that distinction relies on the trusted-writer boundary and still validates native status, exact output size, and output CRC.

## Portable alignment policy

The production encoder uses a 16-byte portable payload alignment. The first payload offset and every later compressed payload offset are rounded up to 16 bytes; padding bytes are zero and are included in the compressed-payload CRC. The format requires the minimal aligned offset after the header or preceding payload, so extra padding is not another valid encoding of the same record. Complete-record validation rejects non-zero gap bytes.

Production uncompressed chunk size is also divisible by 16, except for the final chunk. Given an aligned output base, prefix-sum output ranges are then aligned for every non-final chunk. The final chunk has no following output range whose alignment depends on its size.

Sixteen bytes exceeds nvCOMP raw Deflate's current four-byte requirement and keeps records portable without encoding a vendor name. A backend still checks resolved native addresses against its runtime requirements. Records that do not satisfy a selected backend may be repacked explicitly or rejected; they are never submitted on the strength of caller-asserted alignment metadata.

## Module ownership

PR2 keeps its three accelerator-facing values under `lmcache/v1/distributed/compress_adapters/device.py`:

```text
DeviceIdentity
DeviceBufferRange
DeviceExecutionContext
```

They adapt LMCache's platform APIs, but their current lifetime and validation rules are specific to native decompression. Moving an interface into `platform/base/` would make it a shared platform contract before another consumer has demonstrated the same requirements. A later change may promote a value unchanged when there is a second use.

The remaining compression-specific values also belong under `lmcache/v1/distributed/compress_adapters/`:

```text
DeviceOutputLease
DeviceOutputLeaseManager
GpuDecompressExecutionPolicy
GpuDecompressCapabilities
ValidatedCompressedRecord
GpuDecompressItem
GpuDecompressRequest
GpuDecompressCompletion
GpuDecompressBackend
```

Completion validation is compression-specific because it owns per-chunk status, actual-size, and CRC semantics. It does not belong in a generic platform `DeviceCompletion`.

Vendor implementations live in separate modules:

```text
compress_adapters/nvcomp_backend.py
compress_adapters/hipcomp_backend.py
```

Generic modules do not import vendor modules directly. Resolution follows the existing lazy `DeviceSpec` capability pattern so installations without an optional native library can still import LMCache.

## Shared device values

The three values in this section are implemented and belong to PR2. They do not submit work or claim ownership of an LMCache allocator slot.

### Device identity

```python
DeviceIdentity(device_type, backend_name, device_index)
```

`device_type` is Torch's device spelling. `backend_name` comes from the selected LMCache `DeviceSpec`, so CUDA and ROCm remain distinct even though both use Torch's `cuda` device type. `device_index` is always concrete; an index-free accelerator device is resolved through the selected Torch runtime. Equality requires every field to match.

Resolving an implicit accelerator index may initialize the runtime. Exceptions from that query propagate unchanged through both device-identity and execution-context construction, so callers can distinguish runtime failures from invalid arguments. A backend without a callable current-device query still raises `ValueError` for an index-free device.

### Device buffer range

```python
DeviceBufferRange.from_tensor(
    tensor,
    byte_offset=byte_offset,
    byte_length=byte_length,
)
```

The PR2 buffer adapter deliberately accepts only a contiguous `torch.Tensor`. It rejects packed quantized dtypes (`torch.quint4x2` and `torch.quint2x4`) because their logical element count times element size overstates physical storage. It derives device identity, capacity, and native address from that exact tensor view; callers cannot supply those claims independently. The represented range may be empty, but it must stay inside `tensor.numel() * tensor.element_size()`. `MemoryObj` support is deferred because retaining its Python object does not prevent the allocator slot from being freed or reused.

Construction validates `byte_offset + byte_length` with overflow-safe arithmetic. A vendor backend repeats capacity, device, live address, and actual `(pointer + offset)` alignment validation immediately before submission. This second validation also detects aliasing between distinct tensor views of one allocation.

The value retains its tensor but is only a range descriptor, not an allocation lease. PR2's output lease reserves the range against other acquisitions through its manager; it cannot prevent external tensor mutation or staging use. Callers must not resize or rebind the reserved tensor or its storage while the lease is active, and must release only after all GPU work accessing it has completed or been drained. PR3 will define the submission and completion ownership handoff.

Equality and hashing use the snapshotted device, address, capacity, offset, and length. The retained tensor is excluded from comparison because Torch equality is element-wise, and it is excluded from representations so logging a range does not print KV contents.

### Device execution context

```python
DeviceExecutionContext.from_cache_context(cache_context)
```

The adapter captures the existing cache context's device and opaque stream owner, typed as `object`. It does not add another abstract property that every CPU or accelerator cache context must implement. Only the matching vendor backend interprets the owner's concrete type and translates it to `cudaStream_t`, `hipStream_t`, or an equivalent native type.

A backend submits on the supplied context and never switches silently to a global or default stream. Input H2D work is enqueued on the same context. The adapter retains the context, but cannot prevent another thread from closing it; production shutdown must quiesce decompression handlers before closing cache contexts or backend resources.

## Output staging leases

```python
lease_manager = DeviceOutputLeaseManager.for_execution_context(execution_context)
output_lease = lease_manager.acquire(
    batch_idx=batch_idx,
    object_group_idx=object_group_idx,
    byte_length=expected_uncompressed_size,
)
```

The context-keyed factory returns one live thread-safe reservation authority for the compressed-output staging ranges of each cache context; direct manager construction is rejected. Cache contexts must support weak references and retain `object` identity equality and hashing; contexts that override either are rejected before registry lookup so distinct owners cannot share an authority or a closed-state entry. The manager derives each range from `cache_context.get_temp_object_group_buffer()` rather than accepting a caller-provided tensor, verifies that it belongs to the execution-context device, and rejects physical overlap with any active lease. Address-based overlap catches aliases created from different tensor views; adjacent half-open ranges are permitted, and empty ranges reserve no writable byte. Each lease retains its `batch_idx` and `object_group_idx` for later placement.

Object-group leases always start at byte offset zero because LMCache's existing paged-placement kernels read from fixed kernel-group bases within that staging slot. `DeviceBufferRange` remains offset-capable for other native-buffer uses, but an offset output would require a separate offset-aware placement contract.

`DeviceOutputLease` is an identity-bearing capability, not another value descriptor. Its manager retains it until explicit release, so dropping a caller reference cannot silently make the range reusable. Release is idempotent but is valid only after every GPU operation that may access the range has completed or been drained. A later `GpuDecompressCompletion` owns that sequencing.

Managers with active leases are strongly retained outside the weak context registry. If callers lose every reference to an active capability, the reservation therefore remains unavailable rather than being reclaimed by cycle collection. Active capabilities and their owners remain retained until explicit release or process exit; closing the cache context does not reclaim them, and manager closure rejects active leases. This fail-safe retention avoids reusing bytes that GPU work may still address. Shutdown draining and resource finalization belong to PR3.

The manager does not intercept code that bypasses it. Existing raw transfer paths reuse fixed `batch_idx` staging slots through same-stream ordering rather than a standalone allocator. Production integration must therefore install one manager per cache context and either serialize raw and compressed staging use or route both through the same reservation authority. Closing a manager with active leases fails instead of abandoning them.

## Decompression request

One `GpuDecompressItem` joins one trusted host input with the exact staging destination required by the caller's logical KV layout:

```python
GpuDecompressItem(
    validated_record=validated_record,
    output_lease=output_lease,
    expected_uncompressed_size=expected_uncompressed_size,
)
```

- `validated_record` proves that the complete immutable host record has exact length and a valid header plus compressed-payload checksum. The backend later uploads these exact retained bytes, so a caller cannot pair validated metadata with unrelated device input.
- `expected_uncompressed_size` comes from the caller's logical KV layout and must exactly equal `validated_record.header.uncompressed_size`; stored metadata cannot choose an unbounded or differently sized output.
- `output_lease` reserves a context-owned `DeviceBufferRange` whose `byte_length` must exactly equal `expected_uncompressed_size`; no native backend is involved in establishing this size equality.
- Chunk input ranges come from descriptor payload offsets and compressed sizes.
- Chunk output ranges are consecutive prefix-sum ranges in descriptor order.
- The backend modifies no byte outside an item's exact output range.

Every request carries explicit caller-selected operational ceilings:

```python
GpuDecompressRequestLimits(
    max_records=max_records,
    max_compression_chunks=max_compression_chunks,
    max_total_record_bytes=max_total_record_bytes,
    max_total_compressed_payload_bytes=max_total_compressed_payload_bytes,
    max_total_uncompressed_bytes=max_total_uncompressed_bytes,
)
```

The generic layer intentionally supplies no universal defaults because safe limits depend on the caller's workload and the eventually selected device, vendor library, and execution engine. Request limits provide an early backend-independent bound; a selected backend may advertise and enforce stricter capability limits.

One `GpuDecompressRequest` contains a non-empty tuple of items and the limits applied to that batch:

```python
GpuDecompressRequest(items=items, limits=limits)
```

Construction requires every item to use one live `DeviceOutputLeaseManager`, rejects duplicate lease capabilities, and requires one homogeneous `(record version, codec, framing, transform)` across the batch. The single-manager rule means one request maps to one cache context, device, and stream snapshot; it cannot accidentally combine destinations governed by independent reservation authorities.

The request separately exposes and caps `record_count`, flattened `compression_chunk_count`, `total_record_bytes`, `total_compressed_payload_bytes`, and `total_uncompressed_bytes`. Complete record bytes include headers, alignment gaps, and compressed payload because that is the exact host-to-device input footprint; compressed payload bytes count only descriptor-declared streams because that is a distinct codec-work metric. Names remain distinct in errors and future metrics.

Lease activity is time-varying even though the item and request values are frozen. Construction verifies that every lease is active, and `validate_active_leases()` provides the same check at the later submission boundary. The caller must still obey the lease lifecycle contract and not release a reservation until all GPU work and final validation have completed.

Empty records are valid items. They retain a zero-length lease, count as records and complete host headers, but contribute no compression chunks, compressed payload bytes, or output bytes. A later backend returns an already successful completion for an all-empty request without calling a vendor API.

## Backend validation and capabilities

Backend and context validation cannot occur in request construction because a request contains neither. The backend exposes:

```python
backend.validate_request(request, execution_context, execution_policy)
```

It validates:

- format/version, runtime, device, library, and execution-policy support;
- leased output and context device identity;
- output owner types and resolved native ranges;
- actual output pointer alignment;
- output/output non-aliasing within the request and across active submissions;
- chunk, record, aggregate-byte, and active-submission limits;
- backend- and engine-specific chunk-size limits.

Device inputs do not exist during this advisory validation. `submit_batch()` allocates them from the backend-owned compressed-input pool, validates their live addresses, alignment, and non-aliasing with every leased output, and then enqueues the exact H2D copies. Every leased output is disjoint from all other output and input ranges in active submissions.

`SupportedCompressedRecordFormat` makes each supported `(record version, codec, framing, transform)` identity explicit and hashable. `GpuDecompressCapabilities` is an immutable profile for one concrete `DeviceIdentity`, not a vendor-wide promise across heterogeneous GPUs. A future backend resolves the profile for the supplied execution context so device- and engine-specific limits remain accurate.

The profile reports `decompress_backend_name` (for example `nvcomp` or `hipcomp`) separately from `device.backend_name` (for example `cuda` or `rocm`), plus the installed library name and opaque version, conservative stable or experimental status, supported record identities and execution policies, maximum compressed and uncompressed chunk bytes, maximum records and flattened chunks, separate aggregate complete-record, compressed-payload, and uncompressed-output bytes, maximum active submissions, native input/output/workspace alignments, and asynchronous behavior.

All maximums and alignments are positive, and aggregate payload/output capacities cannot be smaller than their related per-chunk maximum. The complete-record capacity must at least contain the compressed bytes of one maximum-size chunk, but this is only a lower-bound consistency check; later request validation accounts for the selected record version's header and alignment overhead. Per-chunk capability maxima are not globally capped to version 1's uint32 fields because a profile may describe future record versions, while the active parser already enforces the wire bounds of every constructed request. Alignments are not assumed to be powers of two because shared validation can use modulo arithmetic. Limits have no generic defaults: concrete backends derive conservative values for the selected device, installed native library, and execution engine.

Capability query helpers answer whether the profile supports an exact record identity, execution policy, or device. They reject wrong argument types, return `False` for integer record versions outside the unsigned-byte format domain, and otherwise return booleans; typed unsupported-request failures belong to the later backend validation contract.

`is_asynchronous=False` means device work is complete when submission returns; it never exempts a caller from validating or discarding the returned completion. Both synchronous and asynchronous profiles use the same finalization contract so validation results and owned resources are released consistently.

## Execution policy

The execution-policy values are:

```text
any_gpu_engine
software_gpu_required
fixed_function_required
```

`any_gpu_engine` lets the backend choose any supported GPU engine but never permits CPU fallback. `software_gpu_required` requires ordinary CUDA/HIP compute execution and forbids silent fixed-function selection. `fixed_function_required` requires a dedicated hardware decompressor and forbids software fallback.

nvCOMP may use CUDA kernels or Blackwell's fixed-function Decompression Engine. Its default selection may fall back to CUDA kernels. When fixed-function work is required, `NvcompBackend` selects the strict hardware mode and validates device support, dynamically queried chunk limits, and allocation provenance for every item. It fails instead of falling back.

hipCOMP currently uses HIP compute kernels and therefore advertises `any_gpu_engine` and `software_gpu_required`, but not `fixed_function_required`. Its initial Deflate/Gzip profile reports experimental stability while upstream marks those APIs experimental.

## Input, workspace, and concurrency ownership

Compressed device input and native workspace are backend-owned because their layout and lifetime are internal to submission. An active submission leases distinct aligned ranges from both pools. `required_workspace_size()` remains a pure diagnostic and planning operation; callers do not pass the returned allocation to `submit_batch()`.

The backend reserves an active-submission slot, device input ranges, a distinct aligned workspace slot, and native descriptor/status arrays before its first enqueue. It never reuses them while their completion is active. Allocation or backpressure failure occurs before H2D or native submission.

Backend methods are thread-safe. Capabilities report the maximum active submissions; exhausting it raises a deterministic busy error rather than reusing active resources. `required_workspace_size()` is pure and reentrant.

## Backend and completion interfaces

The shared operation has this shape:

```python
backend.validate_request(request, execution_context, execution_policy)
workspace_size = backend.required_workspace_size(
    request,
    execution_context,
    execution_policy,
)
completion = backend.submit_batch(
    request,
    execution_context,
    execution_policy,
)
try:
    completion.wait_and_validate()
finally:
    completion.wait_and_discard()
```

`submit_batch()` repeats backend validation to close races between advisory validation/sizing and submission.

Submission has a strict ownership boundary:

```text
submit raises  -> no device work remains; caller can immediately reuse buffers
submit returns -> completion controls lifetime until finalization
```

If an error occurs after any work may have been enqueued, the backend drains that work and raises only afterward, or returns a terminal-failure completion. It never raises while orphaned work can still access caller buffers.

`GpuDecompressCompletion` exposes:

```python
class GpuDecompressCompletion(Protocol):
    def query(self) -> bool: ...
    def wait_and_validate(self) -> None: ...
    def wait_and_discard(self) -> None: ...
```

Native work is not cancellable initially. `wait_and_discard()` drains work and releases resources without making output usable. Every returned completion is finalized in a `finally` block, including cancellation and validation failure.

`wait_and_validate()` finalizes and releases resources after reaching either a stable success or stable validation failure. The `finally` call to `wait_and_discard()` is an idempotent safety net for host interruption while a wait is in progress.

The completion retains the request and its output leases, the execution-context owner, backend-owned input leases, workspace, descriptors, status arrays, actual-size arrays, and CRC arrays until finalization.

### Completion states

| State | `query()` | `wait_and_validate()` | `wait_and_discard()` |
|---|---|---|---|
| pending | `False` | waits, then validates | waits, then discards |
| device complete, unvalidated | `True` | validates | discards |
| validated success | `True` | returns again | no-op |
| validated failure | `True` | replays stable failure | no-op |
| discarded | `True` | raises discarded-result error | no-op |

`query()` never releases resources and does not imply usable output. Concurrent waits produce one stable terminal state. Backend shutdown atomically stops new submissions, finalizes active work, and then releases resources; repeated shutdown is safe.

## Submission and visibility flow

```text
1. Load the exact record and verify header plus compressed-payload checksums.
2. Acquire output leases and build a homogeneous request from validated records
   and logical KV sizes.
3. Validate the request, output ranges, execution context, policy, and limits.
4. Enter `submit_batch()` and repeat validation while taking backend ownership.
5. Reserve the active-submission slot, device inputs, workspace, and native
   metadata; validate all live addresses and overlap before the first enqueue.
6. Enqueue exact-length H2D copies from each validated record on the selected
   execution context.
7. Upload descriptors and submit all non-empty chunk streams on that context.
8. Compute output CRCs on the same context.
9. Make status, actual-size, and CRC results host-readable.
10. Record completion only after every validation result is ready.
11. Validate every status, output size, and CRC.
12. Only then enqueue or expose paged-KV placement.
```

The initial implementation may synchronize the host during validation for correctness. A later implementation may make validation stream-ordered, but it cannot weaken the visibility rule.

## Failure contract

Failures before submission enqueue no work. Failures after ownership transfer are represented or drained before caller resources become reusable.

Typed failures distinguish invalid requests, unsupported formats or policies, backpressure/submission failures, native execution failures, and validation failures.

Any item failure invalidates the complete staging request. Failed or discarded output is indeterminate and never placed. CPU fallback is permitted only through an explicit policy outside this backend interface.

A `GpuDecompressRequest` corresponds to one reuse-safe staging batch, not an entire retrieve. A retrieve can contain several staging batches and a mixture of raw and compressed records. If a later batch fails after earlier placement, the overall retrieve returns failure and its exported completion event does not authorize use of any partially written blocks; the caller recomputes. This is logical atomicity, not physical rollback of prior device writes.

## L1 and L2 integration prerequisites

These changes are later milestones, but their contracts must exist before the backend is connected to production.

### Variable-size loads

An L2 load destination is a capacity, not a required exact stored length. An adapter succeeds when the stored object fits, writes the exact bytes, and calls `MemoryObj.set_used_size(actual_bytes)`. It fails without truncation when the stored object exceeds physical capacity. Existing fixed-size loads continue to set an actual size equal to capacity.

This preserves the existing load bitmap while allowing a serde wrapper to allocate an upper bound, load a smaller compressed record, and then parse only its exact used bytes. Adapters such as filesystem storage that currently require file size to equal destination length must adopt the capacity/actual size rule.

### Representation identity

L1 state gains a public representation value, initially:

```text
raw_kv
portable_compressed_record
```

Representation metadata is stored with the L1 object state and exposed through a public accessor; retrieve code never reads private metadata. Raw and compressed values do not coexist under one logical `ObjectKey` initially. An existing raw L1 hit wins; an object loaded from compressed L2 is tagged as a portable record before becoming read-ready.

The record becomes read-ready only after its exact length, header, and compressed-payload checksum are validated. A failed record is removed. Existing read locks, eviction, cancellation, and stream-ordered release semantics remain unchanged.

## Relationship to existing LMCache components

- `GPUCacheContext` already owns an opaque stream and flat per-object-group staging tensors. The output buffer wrapper derives from those tensor views.
- The decompression backend owns a compressed-input pool and leases exact record-sized ranges to active submissions; output reuses existing raw-KV-sized context staging.
- `SerdeL2AdapterWrapper` later preserves a validated portable record in L1 instead of always CPU-materializing it.
- The multiprocess retrieve path classifies raw and deferred representations, orders compressed H2D work, invokes the backend, validates completion, and then uses the existing paged-placement operation.
- Existing stream callbacks retain and release L1 locks only after all decompression, validation, and placement work is ordered correctly.

## Delivery plan

Deliver the feature in seven PRs. PR2 packages the existing shared contracts; PR3 adds the unfinished asynchronous backend and completion lifecycle. Keeping those reviews separate lets the implemented value and reservation contracts proceed without waiting for submission, finalization, and shutdown behavior. Each PR includes its own tests and relevant documentation.

| PR | Scope | Implementation status | Depends on |
|---|---|---|---|
| 1 | Portable record format and validation, reference codec, frozen fixtures, and standalone NVIDIA/AMD compatibility probes | Packaged on `feat/gpu-decomp-record-format`; Python tests pass. NVIDIA compatibility evidence is recorded; AMD hardware validation remains pending. | None |
| 2 | Shared device identity, buffer and execution-context adapters, output leases, bounded requests, execution policies, and capability profiles | Packaged on `feat/gpu-decomp-contracts`, including the reviewed validation and ownership fixes. | PR1 |
| 3 | Backend interface, typed backend failures, completion lifecycle, and fake CUDA/HIP backend conformance tests | Not implemented. | PR2 |
| 4 | `NvcompBackend`, optional dependency resolution, native submission, resource ownership, and output validation | Not implemented. The PR1 probe is not a production backend. | PR3 |
| 5 | `HipcompBackend` implementing the same contract with explicit experimental feature gating | Not implemented. The PR1 probe is not a production backend. | PR3 |
| 6 | Storage integration: variable-size L2 loads, portable-record serde, and validated compressed L1 representation | Not implemented. | PR1; coordinate the input-ownership handoff with PR3. |
| 7 | Retrieve integration, configuration, paged placement, and end-to-end correctness and performance validation | Not implemented. | PR4, PR5, PR6 |

PR4 and PR5 can develop independently after PR3. Storage integration can proceed alongside backend work; activation waits for the retrieve path. Both vendor implementations remain explicit deliverables. The remaining implementation boundaries may be adjusted when their review size is known, while preserving these responsibilities and dependencies.

### Review and validation boundaries

- **PR1:** Verify wire bounds, checksums, canonical alignment, immutable ownership, exact logical output size, stream termination, and frozen fixtures. Keep native compatibility evidence separate from production-backend readiness.
- **PR2:** Cover owner-derived capacity and identity, aliasing, device/context mismatch, exclusive leases, homogeneous batching, request limits, and capability consistency. Include no backend submission or completion state machine.
- **PR3:** Cover advisory versus submission-time validation, output status/size/CRC failures, partial enqueue, resource isolation, backpressure, concurrent finalization, shutdown races, and stable failure replay. Define how staging stays protected through subsequent placement. Keep real vendor imports and production call sites absent.
- **PR4 and PR5:** Apply the same conformance suite to each concrete backend and validate on the corresponding hardware. Enforce advertised engine policies and capability limits without silent fallback.
- **PR6:** Cover exact used sizes, oversized-load rejection, representation publication, corrupt-record removal, raw-hit precedence, and input ownership through cancellation or eviction. Choose an immutable-copy baseline or implement the guarded L1 input form explicitly.
- **PR7:** Cover mixed raw/compressed retrieval, staging reuse through placement, cancellation, later-batch failure, and shutdown. Preserve the existing raw transfer paths. Scope initial retrieval to the whole-chunk path; coordinate layerwise layout and readiness semantics before adding layerwise support.

All PRs target `dev`. Dependent drafts may be reviewed concurrently, but their diffs include unmerged prerequisites until those prerequisites land and the branches are rebased. PR2 can be packaged from the existing code without treating PR3's unimplemented lifecycle as completed work.

## Vendor constraints informing the contract

- nvCOMP warns that corrupt Deflate/Gzip input can produce undefined behavior; this motivates pre-native integrity and trust validation.
- nvCOMP raw Deflate currently requires four-byte alignment, motivating the stricter portable 16-byte policy.
- nvCOMP's default engine may fall back from fixed-function hardware to CUDA kernels, motivating explicit execution policy.
- hipCOMP 2.3 has compatible batched asynchronous Deflate/Gzip concepts but labels them experimental and unsuitable for production workloads. The RFC still includes a feature-gated implementation, which reports that stability honestly in capabilities.

Primary references:

- [nvCOMP Deflate/Gzip C API](https://docs.nvidia.com/cuda/archive/13.1.0/nvcomp/c_api.html)
- [NVIDIA Decompression Engine FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)
- [hipCOMP 2.3 algorithm support](https://github.com/ROCm/hipCOMP-core/blob/22cc762f54fba7cdfca74a4c50c00f2aac4ace7a/README.md#algorithm-support)
- [hipCOMP Deflate API](https://github.com/ROCm/hipCOMP-core/blob/22cc762f54fba7cdfca74a4c50c00f2aac4ace7a/include/hipcomp/deflate.h)
- [hipCOMP Gzip API](https://github.com/ROCm/hipCOMP-core/blob/22cc762f54fba7cdfca74a4c50c00f2aac4ace7a/include/hipcomp/gzip.h)

## Decisions deferred until compatibility probes

- Raw Deflate versus Gzip as the production framing.
- Production compression chunk size, subject to the 16-byte alignment policy.
- Initial CPU/QAT/IAA store-side encoder.
- Pinned nvCOMP and hipCOMP versions and exact common capability set.
- Exact feature-gating and user-facing stability language for the initial hipCOMP backend.

These choices do not weaken the integrity, ownership, submission, or completion semantics above.
