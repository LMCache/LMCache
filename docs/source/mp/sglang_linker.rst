SGLang Unified Cache Linker
===========================

``LMCacheLinker`` is an out-of-tree implementation of SGLang's
``UnifiedCacheLinker``. SGLang owns prefix matching, GPU allocation, tree locks,
eviction, and recurrent-state copy-on-write. The plugin owns LMCache MP
registrations, remote read locks, and asynchronous transfers. It does not
introduce an LMCache-specific radix tree.

Requirements
------------

Use an SGLang revision containing the external linker module loader and the
generic page-buffer/Mamba linker interfaces. The draft PR's validation section
pins the tested SGLang and LMCache revisions; released SGLang versions without
these interfaces cannot load this plugin. Both the MP server and the SGLang
workers must install this LMCache revision, including its native CUDA extension.

The current transfer path requires CUDA IPC and contiguous device buffers in
NHD layout. It supports full attention (MHA/MLA), sliding-window pools, and
unquantized Mamba checkpoints. Quantized Mamba checkpoints and standard-pool
MTP draft buffers are rejected. Specialized DeepSeek layouts use SGLang's
existing pool assembler; model-level validation is separate from byte-transfer
coverage. CPU and non-CUDA devices are not supported by this plugin.

The MP process must have access to the same physical GPUs and IPC namespace as
the workers. Tensor-parallel ranks register separate objects; the SGLang tree
intersects restorable boundaries across ranks. Model path, revision, page
layout, pool name, and TP/PP/CP coordinates isolate cache identity. Set a new
``namespace`` when weights or runtime transformations change without changing
the model revision. LoRA and request salts remain part of SGLang's page keys.

Start the services
------------------

Run a dedicated MP server with **chunk size 1**. Here, one logical MP chunk is
one complete SGLang page or recurrent checkpoint, not one language-model token.
Server token counters therefore count these logical units. These objects are
not interchangeable with caches written by the vLLM connector.

.. code-block:: bash

   python -m lmcache.v1.multiprocess.server \
     --host 127.0.0.1 --port 5555 \
     --chunk-size 1 --l1-size-gb 4 --eviction-policy LRU

In a second terminal in the same environment:

.. code-block:: bash

   python -m sglang.launch_server \
     --model-path Qwen/Qwen3-0.6B \
     --revision c1899de289a04d12100db370d81485cdf75e47ca \
     --host 127.0.0.1 --port 30000 \
     --tp 1 --page-size 16 --attention-backend triton \
     --disable-cuda-graph --mem-fraction-static 0.15 \
     --max-total-tokens 4096 --enable-cache-report \
     --enable-unified-cache-external-linker \
     --unified-cache-external-linker-config '{
       "linker": "LMCacheLinker",
       "linker_module_path": "lmcache.integration.sglang.unified_cache_linker",
       "linker_extra_config": {
         "server_url": "tcp://127.0.0.1:5555",
         "namespace": "qwen3-demo",
         "timeout": 60,
         "heartbeat_interval": 10
       }
     }'

The module is imported in cache-owning workers. No LMCache backend name is added
to SGLang. Do not combine this configuration with the legacy ``--enable-lmcache``
integration or hierarchical cache mode.

``timeout`` and ``heartbeat_interval`` are positive values in seconds. Unknown
configuration keys and a server chunk size other than 1 fail at initialization.
A lost server registration or a failed required load fails closed; the plugin
does not acknowledge invalid KV as reusable.

Ownership and transfer order
----------------------------

.. mermaid::

   flowchart LR
     Tree["SGLang unified radix tree<br/>prefixes, slots, locks, eviction"]
     Linker["LMCacheLinker in worker<br/>lookup leases, IPC registrations, futures"]
     MP["LMCache MP process<br/>L1 / configured L2 storage"]
     GPU["SGLang GPU buffers<br/>KV pages and recurrent checkpoints"]
     Tree -->|"keys + component transfers"| Linker
     Linker -->|"lookup / store / retrieve"| MP
     MP <-->|"CUDA IPC copies"| GPU
     Linker -->|"completion after DMA"| Tree

Lookup reserves matching remote objects without allocating GPU slots. Full
attention requires a contiguous prefix. SWA and Mamba probe candidate boundaries
independently, so a missing earlier checkpoint cannot hide a later valid one.
Metadata lookup waits for MP prefetch completion; this implementation does not
provide scheduler-overlapped lookup.

SGLang selects a boundary and allocates destinations. The plugin submits GPU
retrieves and releases unused lookup locks. Its layer counter waits on one
all-layer completion event per transfer; the storage format is not required to
be layerwise. Forward computation is ordered with a stream wait, while tree
locks remain held until the actual GPU copy completes. Stores also wait for
the producer stream and finish before their source slots can be recycled.

A restored Mamba checkpoint belongs to the radix tree. The request receives a
separate mutable state slot and uses SGLang's deferred copy-on-write. Cancelling
a request releases unused lookup locks, while a load already published into
the tree completes before its destination lock is released. Flush drains GPU
work and clears local state while preserving remote objects. Close drains
transfers, releases read locks, and unregisters device memory. Transport
timeouts retain uncertain operations rather than double-unlock shared objects
or recycle memory that might still be used by DMA.

Validate an external hit
------------------------

From the LMCache repository root:

.. code-block:: bash

   python examples/sglang/kv_linker/verify.py --base-url http://127.0.0.1:30000

The verifier uses a fresh cache salt, checks a cold miss, records a native local
radix hit, flushes only SGLang's local radix cache, and repeats the prompt. It
requires positive ``host`` cached tokens, zero ``device`` cached tokens, equal
local/external prefix lengths, and identical greedy output tokens for all three
requests. The external restore's log probabilities must differ from the native
local hit by less than 0.05. The cold-prefill difference is reported separately:
different prefill shapes can change floating-point results even without an
external transfer. In linker mode, SGLang's ``host`` counter includes direct
external loads.

For real MP/GPU byte-transfer checks without model weights:

.. code-block:: bash

   LMCACHE_LINKER_TEST_SERVER=tcp://127.0.0.1:5555 \
     pytest -q tests/v1/test_sglang_linker_mp.py

These tests close the writer registration and restore into fresh GPU buffers
and different page slots. They cover MHA/MLA-shaped buffers, sparse SWA windows,
Mamba endpoints, multiple planes, and mixed dtypes. They do not by themselves
establish model-level correctness for every attention architecture.
