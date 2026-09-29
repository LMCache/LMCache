Per-chunk store completion
==========================

The optional chunk store module lets an asynchronous offload consumer reclaim
source GPU blocks as each token chunk finishes copying. Enable it on the MP
server with:

.. code-block:: bash

   lmcache server --enable chunk_store

Use ``TransferContext.submit_store_with_chunk_events`` or the ATOM adapter's
``submit_store_request_with_chunk_events``. The worker discovers whether the
server enables the module. When it is disabled, these entry points return an
ordinary store future, so consumers must always support terminal completion.

When the returned future supports ``take_completed_ranges()``, that method
returns newly completed absolute token ranges without blocking. Each range is
returned at most once. It indicates that the corresponding source buffers are
safe to reuse; it does not indicate independently retrievable cached data.
``query()``, ``wait()`` and ``result()`` still refer to the whole store.

Keep the transfer context alive until its futures complete and close it before
closing the request client. The context drains completion-event release
acknowledgements before unregister and on close. Losing worker liveness
invalidates outstanding handles; discard old futures before reconnecting.

The chunk future supports host polling and does not expose device-stream waits.
Both ZMQ and gRPC transports are supported. Per-chunk transfer calls and events
add overhead, so use this entry point when early source-block release is useful.
