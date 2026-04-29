Non-KV caching
==============

Most of LMCache deals with the **KV cache** — the per-token attention
key/value tensors produced during prefill and reused during decode.
But LLM serving also performs other heavyweight, request-scoped
computations whose outputs are equally cacheable. LMCache reuses the
same storage backends (local CPU, local disk, remote, NIXL, …) to
cache those outputs as well, under the umbrella of *non-KV caching*.

What's in this section
----------------------

* :doc:`encoder_cache/index` — caches the **vision encoder output**
  for multimodal models (vLLM v1's encoder cache connector). When the
  same image or video is referenced by two requests, the second
  request reuses the cached encoder tensor and the vision tower does
  not run again.

When to consider non-KV caching
-------------------------------

Non-KV caches are most useful when:

* The same multimodal input (e.g. an image, a video) appears in many
  requests over time — for example a product catalog image used in
  many shopping queries, or a manual page used in many support chats.
* The non-KV computation is a non-trivial fraction of prefill time.
  For multimodal models this is true at moderate-to-large
  ``num_frames`` / image resolution; for short images with long text
  prompts the win is smaller.
* You want sharing across processes or restarts. Like KV caching,
  non-KV caches survive process restarts when stored on disk or remote
  backends, and they are shared across tensor-parallel ranks.

How non-KV caching relates to KV caching
----------------------------------------

The KV cache and any non-KV cache are owned by **separate
StorageManager instances** — they cannot evict each other. This is
intentional: the access patterns are different (KV is chunked,
layerwise, high-volume; non-KV is single-tensor, request-scoped) and
keeping the pools separate makes resource budgeting auditable. Each
non-KV cache exposes its own configuration prefix (e.g.
``LMCACHE_EC_*`` for the encoder cache), so an operator can size
local CPU / disk independently per cache type.

.. toctree::
   :maxdepth: 2
   :caption: Non-KV caches
   :hidden:

   encoder_cache/index
