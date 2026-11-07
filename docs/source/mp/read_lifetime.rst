Experimental read lifetime
==========================

The ordinary vLLM ``LMCacheMPConnector`` can opt into named lookup reservations
and completion-based source-memory protection. Use the same LMCache build on
the vLLM client and every MP server.

Add these entries to your existing ``kv_connector_extra_config``:

.. code-block:: json

   {
     "lmcache.mp.read_lifetime_v1": true,
     "lmcache.mp.mp_transfer_mode": "lmcache_driven"
   }

The feature is disabled by default. The scheduler checks every server for
read-lifetime protocol version 1 before submitting lookups. Workers reject an
incompatible transfer mode instead of reverting to anonymous read locks.

Each lookup receives a unique identity. A queued reservation may expire and
produce a retrieve miss, allowing recomputation. Once retrieval acquires its
source objects, those allocations remain protected until a device completion
event confirms the copies finished. Request cancellation releases unclaimed
shares while already submitted copies retain their own holds.

This first version supports ordinary MP retrieval, including per-group windows
and TP reader shares. CacheBlend, engine-driven transfers and other inference
engine connectors retain their existing behavior and should not enable this
option. Completion polling adds one device event per transfer and may delay
reclamation by approximately one polling interval (5 ms), plus scheduling
delay. Serving throughput has not yet been benchmarked.

If a device cannot complete or be drained, active memory remains retained.
Shutdown drains pending streams before storage is destroyed. A read TTL is
never used to free memory that a device may still access.
