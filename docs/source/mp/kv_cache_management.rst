.. _mp_kv_cache_management:

KV Cache Management
===================

The multiprocess (MP) server exposes node-local cache inspection and management
endpoints for operators and debugging tools. See :doc:`http_api` for the full
request and response contracts.

Node-local L1 object download
-----------------------------

``POST /cache/objects/download`` downloads one L1 object from the MP server that
receives the request. Callers must already have its exact ``EncodedObjectKey``;
the endpoint does not convert tokens to keys or route through a coordinator.

The response body contains raw logical bytes and the
``X-LMCache-Object-Metadata`` header describes their ordered shapes and dtypes.
Only DRAM and Device-DAX downloads are supported in this version. GDS/AIS-based
downloads return HTTP 501; support for these paths may be added in a future
release. L2, remote-node routing, and batch downloads are not supported.

Raw KV contents can contain sensitive information. Download is **disabled by
default**; explicitly pass ``--enable-l1-cache-download`` to the HTTP server to
enable it. Deploy the MP HTTP interface on a **trusted network**. This flag does
not provide authentication. The default limits are 64 MiB per object and two
concurrent downloads per HTTP process; see :doc:`http_api` for configuration.

The server holds an inspection pin on the source allocation while copying it
into an independent snapshot, then releases the pin before sending the HTTP
response. The pin does not expire with the normal read-lock TTL. Ordinary
deletion and eviction cannot reclaim the object; force deletion removes the key
but defers freeing the allocation until copying finishes. This inspection does not
update LRU state or count as a normal cache retrieve. The snapshot is rejected
if the object is removed or replaced during copying.

Legacy management
-----------------

Operations that are not yet available on the MP HTTP surface, such as general
``move``, ``pin``, and ``compress``, remain documented in the
:doc:`Legacy KV Cache Management <../legacy/kv_cache_management/index>`.

