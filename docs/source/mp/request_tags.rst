Per-request Cache Identity Tags
===============================

MP vLLM requests can add tags to a cache key. Use them when identical token
sequences must not share KV cache entries, for example across application
versions or tenants. Tags supplement ``cache_salt``: every tag is part of the
``ObjectKey``, so a lookup only reuses an entry when its complete tag set
matches the entry that was stored.

Experimental support boundary
-----------------------------

This is an experiment, not yet an all-backend feature. Tagged requests require
every configured L2 adapter to support tags. Currently, only the pure-Python
``fs`` adapter does; ``fs_native`` and other adapters remain unsupported.
LMCache rejects a tagged store or L2 lookup when an unsupported adapter is
configured, rather than allowing two tagged keys to map to the same object.

vLLM request format
-------------------

Pass tags through vLLM's ``kv_transfer_params``. Each identity-bearing setting
must start with ``lmcache.tag.``; the suffix is the tag name and the value must
be a string. Other ``kv_transfer_params`` settings continue to be forwarded but
do not affect cache identity. Supply exactly the same tags on every request
that should reuse an entry.

.. code-block:: python

    client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        extra_body={
            "kv_transfer_params": {
                "lmcache.tag.tenant": "tenant-a",
                "lmcache.tag.application_version": "v2",
            }
        },
    )

Tag names and values are limited to 128 characters and cannot contain ``@``,
``%``, ``/``, ``\\``, or NUL. LMCache sorts tags before using them in key
identity, so request-map order does not matter.

Control-plane request format
----------------------------

The node and coordinator cache-control APIs (prefetch, pin/unpin, delete, and
directory lookup) accept ``tags`` as a JSON object without the internal prefix:

.. code-block:: json

    {"tags": {"tenant": "tenant-a", "application_version": "v2"}}

These APIs resolve the object key using the same rules as the vLLM request
path. The tags must match the stored entry exactly. See :doc:`http_api` and
:doc:`coordinator` for endpoint-specific request bodies.

Storage compatibility
---------------------

For the current ``fs`` experiment, tags are encoded in the file name. Untagged
keys keep the pre-existing filename shape, so existing untagged cache files
remain readable. Untagged and tagged requests are distinct identities and do
not match each other.
