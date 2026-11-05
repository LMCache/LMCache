RESP (Redis/Valkey)
===================

An L2 adapter backed by a native RESP (Redis Serialization Protocol)
connector, targeting **Redis** or **Valkey** servers. I/O is dispatched
through a C++ worker-thread pool.

**Fields:**

- ``host`` (string): Redis/Valkey server hostname or IP. Required unless
  ``LMCACHE_RESP_HOST`` is set.
- ``port`` (int, ``> 0``): Server port. Required unless ``LMCACHE_RESP_PORT``
  is set.
- ``num_workers`` (int, default ``8``, ``> 0``): C++ worker threads for I/O.
- ``username`` (string, default ``""``): Auth username.
- ``password`` (string, default ``""``): Auth password.
- ``max_capacity_gb`` (float, default ``0``): Max L2 capacity in GB for usage
  tracking / aggregate eviction. ``0`` disables tracking.

When ``host``, ``port``, ``username``, or ``password`` are left empty, the
adapter falls back to the corresponding environment variables at creation
time: ``LMCACHE_RESP_HOST``, ``LMCACHE_RESP_PORT``, ``LMCACHE_RESP_USERNAME``,
``LMCACHE_RESP_PASSWORD``. The adapter then rejects the config only when
both the JSON field and the matching env var are missing.

**Configuration examples:**

.. code-block:: bash

    # Basic Redis/Valkey
    --l2-adapter '{"type": "resp", "host": "127.0.0.1", "port": 6379}'

    # With auth, more workers, and a capacity cap
    --l2-adapter '{"type": "resp", "host": "redis.internal", "port": 6379, "username": "lmcache", "password": "secret", "num_workers": 16, "max_capacity_gb": 50}'
