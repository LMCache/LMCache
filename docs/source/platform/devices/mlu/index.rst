.. SPDX-License-Identifier: Apache-2.0

Cambricon MLU Backend
=====================

The MLU backend implements the community ``DeviceSpec`` and ``DeviceOps``
interfaces for multiprocess **engine-driven** transfers. It uses
``torch_mlu`` tensor operations to move KV blocks between MLU memory and
canonical host chunks.

Requirements and selection
--------------------------

Install a matching Cambricon driver, Neuware runtime, PyTorch, and
``torch_mlu`` distribution first. Build LMCache's common C++ extensions
against that PyTorch installation, without GPU-specific extensions::

    NO_GPU_EXT=1 pip install -e . --no-build-isolation

Use an environment with the repository's build dependencies installed.
Do not replace the vendor's PyTorch version as part of this installation.

Set these variables before starting the serving engine::

    export LMCACHE_DEVICE_BACKEND=mlu
    export LMCACHE_MP_TRANSFER_MODE=engine_driven

The backend probes ``torch_mlu`` at startup. Explicit selection fails if the
runtime is missing or no MLU is available. A CPU-only LMCache server does not
need ``LMCACHE_DEVICE_BACKEND=mlu``.

Supported transfer contract
---------------------------

The MP serving adapter must pass real per-layer MLU tensors and an accurate
``kv_layout`` hint. The supported layouts are:

* K/V-first NHD: ``[2, num_blocks, block_size, num_heads, head_size]``.
* K/V-first HND: ``[2, num_blocks, num_heads, block_size, head_size]``.
* MLA: ``[num_blocks, block_size, head_size]``.

The vLLM adapter normalizes legacy MLU layout queries to HND, matching
Cambricon's V1 FlashAttention allocation. Explicitly resolved layouts from
newer vLLM versions retain their reported value.

Host chunks use the same token-major layout as other community backends.
Callers must synchronize the transfer stream before consuming host chunks
or releasing their storage. Completion recorders wait synchronously for the
supplied MLU queue before publishing completion.
The zero handle denotes ``torch.mlu.default_stream()`` on the current
device. A nonzero handle borrows the caller's stream; the caller retains
ownership and must keep the stream alive until the call returns.

The MP transfer context owns source and destination tensors until gather
completion. Error recovery and request-level completion tracking belong to
the shared context and worker adapter. This device backend does not modify
those lifecycle mechanisms or establish end-to-end preemption safety.

D2H copies into non-contiguous host slices also block. This preserves
correctness for prefix skips on torch_mlu versions where an asynchronous
strided D2H copy can return incorrect data even after synchronization.

This backend does not add a vLLM-MLU serving connector. In particular, it does
not implement the vendor connector's remote-allocation extensions. A serving
adapter must use the community MP engine-driven interface.
V1 support alone does not establish connector compatibility: the serving
runtime must also provide the vLLM APIs used by the installed LMCache
connector. A passing device transfer test does not validate those APIs.
Serving adapters must finish pending gathers before KV block reuse,
consume flush metadata before retrieval and forward, and wait for pending
host-to-server commits before closing. Request-level completion receipts
remain the responsibility of the shared worker adapter.

Capability limits
-----------------

Raw pointer transfers, the legacy in-process connector, LMCache-driven IPC,
arbitrary host-memory registration, and native transfer kernels are not
supported. Unsupported transfer entry points raise errors instead of
treating MLU addresses as host pointers. ``auto`` selects ``engine_driven``.
This implementation makes no throughput or latency guarantee.

Validation
----------

Run the backend tests in the configured MLU environment::

    LMCACHE_TRACK_USAGE=false python -m pytest \
        tests/v1/platform/devices/mlu -q

The transfer matrix runs on both CPU and MLU. MLU cases skip when no usable
runtime is present; skipped cases are not hardware validation. The tests
exercise layout correctness and MP gather/scatter dispatch, not full model
serving.

For full serving validation, check KV bit equality across store and restore
separately from generated tokens and log probabilities. Compare the warm
request with native prefix reuse at the same hit length: full prefill and
partial prefill may use different numerical paths. Repeated runs and an
explicit synchronization control help distinguish numerical differences
from stream-ordering failures; a log-probability tolerance alone does not
establish transfer correctness.

Also match the batch timeline: co-scheduled requests, remaining query
lengths, and batch order can change the model's numerical path. To isolate
this effect, replay a captured timeline with identical prefix KV in an
engine without an LMCache connector. Compare logits before sampling and
replay fixed logits separately. Greedy decoding can produce different
tokens when batch-dependent rounding changes two nearly tied logits; a
fixed seed does not establish batch-invariant model computation.
