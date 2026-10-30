# MLU engine-driven device backend

## Scope

`MluDeviceSpec` registers `mlu` with the existing discovery mechanism. It
loads the optional `torch_mlu` package only during availability probing, so
CPU-only installations can still import LMCache. `ops_cls` remains lazy to
avoid a cycle while the platform registry is being constructed.

The first supported path is multiprocess **engine-driven** transfer. The
engine owns MLU tensors, gathers their KV data into host chunks, and sends
the chunks to the LMCache server. Retrieval reverses this operation. This
uses the community device interfaces and does not depend on the private
remote-allocation methods invoked by Cambricon's vLLM-MLU connector.

## Data layout and ownership

The backend accepts per-layer K/V-first NHD (`[2, NB, BS, NH, HS]`),
K/V-first HND (`[2, NB, NH, BS, HS]`), and MLA (`[NB, BS, HS]`). H2D reuses
the shared torch block-transfer implementation. D2H fills one chunk on MLU
a layer at a time, then copies it to the host. Temporary tensors are released
on the transfer stream before allocating the next chunk, limiting device
workspace to one chunk and per-layer scratch. Stored chunks retain the
canonical token-major layout: `[2, NL, T, NH * HS]` or `[NL, T, HS]` for MLA.
The serving adapter must supply the actual `kv_layout` hint for HND; tensor
shape alone cannot distinguish the block-size and head-count dimensions.
The vLLM adapter normalizes the legacy MLU layout query to HND because
Cambricon's V1 FlashAttention backend allocates `[2, NB, NH, BS, HS]` even
when the generic vLLM query returns NHD. A resolved `CacheConfig` layout
takes precedence over this legacy normalization.

Layout selection reads `get_spec_class(engine_kv_format)` facts. It requires
one tensor per layer and either MLA or a leading K/V axis, excluding packed
K/V, fused block/token axes, tuple entries, and blocked-scale repacking.
The tensor rank must be three for MLA or five for NHD/HND: a singleton-axis
HND format shares the HND facts but needs different indexing. Both transfer
directions reject unsupported layouts before modifying either operand.

Both paged buffers and chunks must be tensors. The shared fallback treats
integer chunk pointers as CPU addresses, while the LMCache-driven path can
pass device staging addresses. Rejecting these operands before dispatch
prevents an address-space error. No tensor-from-pointer implementation,
memory IPC, event IPC, or cache context is advertised.

Transfers run on the current torch_mlu stream. A partial prefix skip can
leave a strided host destination. On torch_mlu 1.26.1 with torch 2.6.0,
`copy_(..., non_blocking=True)` into such a destination returns incorrect
data even after device synchronization, with both pageable and pinned host
memory. D2H therefore uses a blocking copy for non-contiguous host slices;
contiguous destinations retain asynchronous copies. This exception is
contained in the MLU backend rather than changing other devices' behavior.

The engine-driven caller must synchronize the transfer stream before publishing
host data, as
required by the existing gather/scatter contract. Completion and event
recorders additionally wrap the supplied native queue in `ExternalStream`
and synchronize before publication. Synchronization errors propagate without
publishing a completion. These recorders block; they do not claim native
asynchronous callback support. Queue handles must belong to the current
MLU device and remain alive for the call.

A zero handle is a default-stream sentinel, resolved through
`torch.mlu.default_stream()`. It must not be wrapped as `ExternalStream(0)`:
on torch_mlu 1.26.1 that null queue can synchronize without waiting for the
actual PyTorch default queue. External wrappers do not own the supplied
native queue; the caller keeps its stream alive until the recorder returns.

The shared transfer context owns KV and host buffers while gather is
pending. Recovery after partial enqueue and tracking of all outstanding
stores are shared lifecycle responsibilities. This device backend uses the
existing context interface without changing those mechanisms. Device
transfer validation does not establish end-to-end preemption safety.

Arbitrary host-memory registration retains the base unsupported capability.
Engine-driven transfers can use pageable host tensors. This does not make a
claim about pinned SHM, zero-copy access, direct RDMA, or serving performance.

## Compatibility boundary

The legacy in-process token-transfer APIs fail explicitly. They cannot use
the shared single-layer NHD indexing on MLU HND caches. LMCache-driven mode
is unavailable and `auto` resolves to engine-driven. Adding an IPC stack or
a native pointer-transfer kernel requires separate implementation and tests.

No generic dispatch code, other device backend, or vendor connector is
modified. The vLLM layout boundary supplies the physical MLU layout.
The backend can later be packaged through `lmcache.device_plugins`
without changing its interface; the initial implementation is in-tree.

## Validation

CPU tests cover discovery, optional-runtime failures, capability selection,
pointer rejection, byte-count validation, and publication ordering/failure.
Transfer tests check NHD/HND/MLA, fp16/bf16/fp32, reordered blocks, partial
prefix skips, complete skips, and untouched destination blocks against an
independent reference. The same matrix runs on MLU when available. Hardware
integration tests exercise the public MP gather/scatter functions, including
stream execution, without replacing the device resolver.
Async-context tests also compare fp16 bit patterns, including signed zero
and subnormals, across producer, restore, and consumer streams. They use
24-layer HND caches, remap blocks, drop source references after submission,
and overwrite host chunks as soon as retrieve releases them. Completion
futures must suffice without an extra device synchronization in the test.

For serving accuracy, compare a restored prefix against native prefix reuse
with the same cached tokens and remaining query length. Full prefill and
prefix reuse can execute different numerical paths. Check the actual KV bits
through store and restore, and compare repeated and forced-synchronization
runs, before attributing a log-probability difference to numerical precision.
Match the co-request timeline as well: batch composition, query lengths,
and request order can affect projection rounding even with identical prefix
KV. Replay fixed logits separately from model computation to distinguish
greedy selection from upstream numerical differences. An isolated native
prefix replay with no LMCache connector can establish whether the same
batch timeline reproduces a serving divergence. This does not establish
batch-invariant generation for the inference runtime.

A passing CPU matrix is not a hardware result. Hardware validation must
record torch and torch_mlu versions and include the real MLU test cases.

The preemption boundary must drain deferred gathers before the serving engine
reassigns KV blocks. A flush can return while the host-to-server commit is
still pending: host buffers must remain owned until that commit completes,
and close must wait before releasing the transport. A gated hardware test
holds prepare and commit separately, overwrites the same KV blocks after
flush, and checks that both generations retain their original fp16 bits.

Serving adapters remain responsible for mapping the runtime's preemption
callbacks to the community MP contract and consuming flush metadata before
retrieval or forward can overwrite blocks. A finished-send receipt must
cover every outstanding gather for that request. These request-level
responsibilities are separate from device operations.
