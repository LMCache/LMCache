.. _mp_kv_cache_management:

KV Cache Management
===================

Current MP operations are documented with their server APIs:

- :doc:`http_api`: cache inspection, clear/delete, prefetch, quotas, and
  runtime L2 adapter configuration.
- :doc:`coordinator`: multi-server discovery, lookup, prefetch, pin/unpin,
  and delete operations.
- :doc:`/cli/kvcache`: command-line cache operations.

The old controller's ``move``, ``pin``, and ``compress`` request schemas are
separate APIs, documented in :doc:`/legacy/kv_cache_management/index`.
Their feature set does not imply API compatibility with the MP coordinator;
use the MP reference for the operation and storage tier being managed.
