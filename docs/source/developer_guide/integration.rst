Integration
===========

LMCache acts as a caching middleware that sits between LLM inference engines and storage systems, enabling:

- **Transparent Cache Reuse**: Automatic detection and reuse of previously computed KV caches
- **Performance Optimization**: 3×–10× reduction in time-to-first-token (TTFT) for multi-round conversation and RAG.
- **Resource Efficiency**: Significant GPU cycle savings through cache reuse
- **Scalable Architecture**: Support for single-node and multi-node deployments

Supported Engines
-----------------

LMCache currently supports integration with:

- **vLLM**
- **SGLang**
- **TRT-LLM** (coming soon)

Integration with vLLM
---------------------

When LMCache is integrated with vLLM, the inference pipeline is augmented to lookup and inject cached KV chunks for any reused input content. The sequence diagram below shows how a prompt request flows through vLLM with LMCache:

.. mermaid::

    sequenceDiagram
    participant User as User Application
    participant VLLM as vLLM Engine (LMCache Connector)
    participant Cache as LMCache Storage (CPU/Disk)
    User->>VLLM: Prompt request (LLM query)
    VLLM->>Cache: **Lookup** KV cache for prompt tokens
    alt Cache hit for some tokens
        Cache-->>VLLM: Return cached KV chunk(s)
        VLLM->>VLLM: Inject KV into model's cache (skip recomputation)
    else Cache miss (no KV for tokens)
        Cache-->>VLLM: No cached data (proceed normally)
    end
    VLLM->>VLLM: **Generate** remaining tokens (compute new KV)
    VLLM-->>User: Respond with completion
    Note right of VLLM: *LLM output is returned without waiting for cache storage.*
    VLLM->>Cache: **Store** new KV chunk(s) (async background put)

**Cache Lookup**: Upon receiving a new prompt, vLLM (via the LMCache connector) computes identifiers (e.g. hashes of token sequences) and queries LMCache for matching KV cache chunks. If a cache hit occurs, LMCache retrieves the KV chunk (potentially from CPU or disk) and returns it to vLLM. vLLM then injects these KV tensors into the model's attention cache instead of recomputing them from scratch.

**Inference with Cache**: vLLM proceeds with inference. For any parts of the prompt where no cache was available (cache miss), the model computes new KV values as usual. Cached segments (e.g. previously seen text) are skipped in computation, significantly reducing the time-to-first-token (TTFT) and saving GPU cycles.

**Async Cache Write**: After processing the prompt, any newly generated KV cache chunks (corresponding to this prompt's content) are handed off to LMCache for storage. This put operation is done asynchronously (in the background) so it doesn't delay the response. The response is returned to the user promptly, and LMCache's background tasks will offload the new KV data to CPU, disk, or other backends for future reuse.

SGLang on MUSA
--------------

LMCache supports SGLang on Moore Threads MUSA devices. In-process transfers
use TorchMUSA ``index_select`` and ``index_copy_`` and support:

* non-layerwise MHA with separate K/V layer pools;
* non-layerwise MLA; and
* layerwise MHA.

Layerwise MLA and multiprocess MLA are not supported. For in-process mode,
configure LMCache without ``mp_host``/``mp_port``:

.. code-block:: yaml

   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 20
   use_layerwise: true  # MHA only; use false for non-layerwise MHA or MLA

When SGLang passes a slot mapping for only the uncached suffix, LMCache applies
the prefix offset before gathering or scattering rows.

The multiprocess MUSA path supports MHA with separate K and V pools. It uses
TorchMUSA memory/event IPC and the Torch block-transfer fallback. The handle
path is opt-in and fail-closed; set this variable in both the SGLang worker and
LMCache server environments:

.. code-block:: bash

   export LMCACHE_MUSA_HANDLE_TRANSFER=1

Then configure the standalone server address as usual:

.. code-block:: yaml

   mp_host: 127.0.0.1
   mp_port: 5556

Startup fails with a capability error when the installed TorchMUSA does not
provide compatible tensor IPC and event IPC APIs. LMCache does not substitute
CUDA IPC wrappers for MUSA tensors.
