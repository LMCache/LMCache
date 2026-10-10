Bounded first-rank retrieval
============================

In-process MLA deployments can keep one copy of each KV chunk on the first rank
with ``extra_config.save_only_first_rank`` and broadcast it to the other ranks
at retrieval time. Non-layerwise retrieval streams the cache hit through one
fixed device staging buffer per engine instead of materializing a device copy
of the whole hit, so device memory no longer grows with the length of the hit.

Configuration
-------------

.. code-block:: yaml

   retrieve_buffer_size: 67108864  # 64 MiB, the default
   extra_config:
     save_only_first_rank: true

``LMCACHE_RETRIEVE_BUFFER_SIZE`` is the equivalent environment override. The
value must be positive, must hold at least one KV chunk (``chunk_size`` tokens
of KV data, checked when the engine is created), and must be identical on all
ranks. The buffer never grows: it is the hard upper bound on the device memory
a retrieval stages.

How a retrieval proceeds
------------------------

1. All ranks agree to admit the retrieval with one host collective: every rank
   must hold a staging buffer at least as large as the first rank's, be
   healthy, and be serving the same request. Otherwise every rank returns an
   all-false mask without transferring anything.
2. The first rank acquires the hit chunks exactly as single-rank retrieval
   does: from prefetch results or storage, with contiguous-prefix semantics
   and the usual pinning.
3. It packs consecutive chunks into the staging buffer until the next chunk
   would not fit, then sends a small host header describing the window.
4. Every rank broadcasts the packed bytes into its own staging buffer, writes
   the window into its KV pages through the GPU connector, and fences the
   write before the buffer is reused.
5. Steps 3 and 4 repeat until the first rank sends an empty header. Sources
   are released once the last window has been fenced.

.. code-block:: text

   all ranks:    admission (one host agreement)
   first rank:   acquire chunks -> pack window -> header -> broadcast -> write -> fence
   other ranks:                                   header -> broadcast -> write -> fence
                                        ^                                        |
                                        +------------ reuse the buffer ----------+

Memory
------

The buffer is allocated when the worker's engine is created and stays resident
for the life of the engine. Compared with the previous implementation, which
allocated device copies per cache hit, this bounds the worst case of a large
hit but adds ``retrieve_buffer_size`` of resident device memory even when
nothing is retrieved.

vLLM creates the KV connector after it has sized its KV pages and before it
allocates them, so the buffer is part of the plan only when vLLM asks the
connector for its device memory requirement first:

- With a vLLM that provides the connector memory hook, LMCache reports
  ``retrieve_buffer_size`` through ``get_required_device_memory`` using the
  same configuration and ``lmcache.*`` overrides as the worker, and vLLM
  subtracts it from the memory available for KV pages.
- With an explicit ``kv_cache_memory_bytes`` budget, the deployment leaves room
  for the staging buffer, model execution, and other workspaces itself.
- With automatic sizing on a vLLM without the hook, the worker starts but logs
  a warning: the buffer comes out of the ``gpu_memory_utilization`` headroom
  and there is no whole-device memory budget guarantee.

Engines in different processes hold separate buffers.

Failures
--------

After admission every rank performs the same sequence of collectives through
every recoverable local failure, so such a failure on one rank never leaves
its peers waiting:

- A missing chunk on the first rank truncates the hit to the contiguous prefix
  before it, on every rank, exactly as in single-rank retrieval.
- A rank whose page write fails fences the work it had already queued, stops
  writing, keeps receiving the remaining windows, and returns the prefix it
  committed. The integration reports the missing tokens to the serving engine
  through its normal load-failure path.
- A storage failure on the first rank, or a chunk larger than the staging
  buffer, ends the exchange for its peers, which return all-false masks,
  before the error propagates on the first rank.
- Device and collective failures propagate and are not cache misses. The
  affected engine marks itself failed and refuses every later retrieval at
  admission, so peers receive all-false results from then on; peers of the
  request in flight may wait until their communicator times out. Terminating
  and restarting the worker group is the serving engine's decision.

Layerwise and multiprocess retrieval are unchanged.
