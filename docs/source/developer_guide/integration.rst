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

LoRA cache isolation
^^^^^^^^^^^^^^^^^^^^

A LoRA adapter changes the attention projections, so the same prompt tokens
produce different KV values under the base model and under each adapter. The
in-process vLLM connector (``LMCacheConnectorV1``) therefore adds the request's
adapter to every cache key as the ``lora`` tag, whose value is a SHA-256 digest
of the adapter's vLLM ``lora_name`` and ``lora_path``. The base model and each
adapter use separate keyspaces even when their prompt token IDs are identical,
and base-model keys are unchanged:

.. code-block:: text

   Qwen/Qwen2.5-0.5B@1@0@1234abcd@bfloat16                 # base model
   Qwen/Qwen2.5-0.5B@1@0@1234abcd@bfloat16@lora%<digest>   # an adapter

The connector sets the ``lora`` tag itself and ignores an ``lmcache.tag.lora``
value sent by the client in ``kv_transfer_params``.

The digest identifies an adapter by its name and path, not by the contents of
its weights. If an adapter's weights change, serve them under a new name or
path. Keeping both can load KV computed with the previous weights from a
persistent backend.

.. warning::

   Earlier versions stored KV computed under an adapter with base-model keys.
   When upgrading, clear persistent backends (local disk and remote storage)
   that may hold entries written by LoRA requests. Otherwise, base-model
   requests can still load them. The in-memory CPU cache starts empty when
   vLLM restarts.

Integration with SGLang
-----------------------

LMCache integrates with SGLang through ``LMCacheUnifiedRadixCache`` and a
standalone LMCache multiprocess server. The legacy in-process, layerwise, and
pre-unified multiprocess connectors are not supported. See the SGLang tab in
:doc:`../getting_started/quickstart` for installation and startup commands.
