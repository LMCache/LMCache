Bounded first-rank retrieval
============================

In-process MLA deployments can store one copy of KV data with
``extra_config.save_only_first_rank`` and broadcast it to the other ranks.
Non-layerwise retrieval uses a reusable device buffer instead of allocating a
second device copy of the entire cache hit.

Configure the payload staging budget in bytes:

.. code-block:: yaml

   retrieve_buffer_size: 67108864  # 64 MiB, the default
   extra_config:
     save_only_first_rank: true

``LMCACHE_RETRIEVE_BUFFER_SIZE`` provides the equivalent environment override.
The value must be positive and identical across participating ranks. Each token's
complete KV data must fit; larger cache chunks are split along the token dimension.
Mixed-dtype groups may require multiple tokens per slice to preserve alignment.
An incompatible layout or an insufficient budget fails retrieval before issuing
the affected tensor collective.

Memory ownership
----------------

The buffer is reserved when the worker's cache engine is created, so integrations
must create that engine before profiling memory available for their final KV
pool. Engines on the same device **within one process** share the allocation and
serialize its use. They must use the same budget. Separate worker processes need
separate budgets accounted for by the serving engine.

The limit covers the broadcast payload buffer. It does not include the serving
engine's final KV pages, connector workspaces, communication-library allocations,
or the existing CPU cache/prefetch pool. Synchronous source reads use bounded
batches sized from the model's chunk layout, with a minimum of one source chunk.
This preserves concurrent backend I/O. Asynchronous retrieval consumes
already-prefetched objects and releases its references as windows complete.

.. code-block:: text

   read source chunk -> pack bounded window -> broadcast -> write final KV pages
                              ^                                  |
                              +------ reuse after completion -----+

Failure and completion
----------------------

Ranks agree on the reservation, request identity (when provided), and token count
before transferring data. Each window has a readiness agreement and a completion
agreement. The result mask becomes available to the caller only after the full
transfer has finished. A source-read, metadata, packing, or destination-write
failure returns an all-false mask on every participating rank. The integration
must report those missing pages to the serving engine for its normal load-failure
handling; the transport does not itself restart or recompute requests.

Integrations must schedule retrievals in the same order across their collective
group. They may supply ``all_ranks_agree_fn`` to ``LMCacheEngineBuilder.get_or_create``
for an efficient host-side agreement; vLLM uses its TP CPU process group. Without
that callback, the engine gathers votes using object broadcasts, whose rank space
must match ``LMCacheMetadata.world_size`` and ``worker_id``.

Connectors must implement ``synchronize_load()`` so it waits for every stream that
can still read temporary input buffers. The default waits for ``load_stream`` when
present, or fences the device otherwise. This also applies when a write raises
after enqueueing work. Device-fence and collective failures propagate and require
the serving engine's device/communicator recovery; they are not cache misses.
After such a failure, the buffer is disabled and the engine retains source
ownership until shutdown can successfully fence the recovered device.

This change does not add direct storage-to-page DMA or alter layerwise and
multiprocess retrieval.
