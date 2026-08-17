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
DRAM and Device-DAX are supported. GDS, L2, remote-node routing, and batch
downloads are not supported in the first version.

The server copies the object into an independent snapshot under temporary read
protection, releases that protection, and only then sends the HTTP response.
This inspection does not update LRU state or count as a normal cache retrieve.

Legacy management
-----------------

Operations that are not yet available on the MP HTTP surface, such as general
``move``, ``pin``, and ``compress``, remain documented in the
:doc:`Legacy section <../legacy/index>`: :doc:`../kv_cache_management/index`.
