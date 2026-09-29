Chunk-store plugin
==================

The chunk-store plugin exposes per-chunk source-buffer completion through the
existing MP request transport. Enable it on an LMCache-driven server:

.. code-block:: bash

   python -m lmcache.v1.multiprocess.server \
     --l1-size-gb 1 --l1-init-size-gb 1 --eviction-policy LRU \
     --server-module '{"module_path":"lmcache.v1.multiprocess.modules.chunk_store"}'

Add ``--transport grpc`` to use gRPC. The module lives in the LMCache package and
uses the pluggable server-module loader; no separate plugin package is required.

After registering KV caches through ``register_kv_cache``, an explicit RPC caller
can submit a range of complete chunks:

.. code-block:: python

   terminal, chunks, success = client.store_with_chunk_events(
       key, instance_id, block_ids, producer_event_handle
   ).result(timeout=10)
   pending = [
       (backend.import_event(handle, device), start, end)
       for handle, start, end in chunks
   ]
   for event, start, end in pending:
       backend.synchronize_event(event, device)
       # Source blocks for [start, end) can now be reused.

For nonblocking completion checks, use ``backend.query_event(event)``. Ranges are
absolute and end-exclusive. Keep the producer event alive until submitted work
completes. Returned handles follow the ordinary store's backend lifetime
constraints, including its bounded retention of exported events; this plugin
does not provide a lease for arbitrarily delayed imports.

``success`` means all chunk stores avoided a fatal error, not that every chunk
was cached. A failure stops later chunks; earlier chunks may already be stored.
The terminal handle covers submitted work and is empty if none was submitted.
Source completion does not guarantee publication has finished. Engine adapters
continue to use ordinary store unless the caller explicitly uses this RPC.
