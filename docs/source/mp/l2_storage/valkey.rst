Valkey
======

An L2 adapter backed by the official ``valkey-glide`` **sync** Python
client. Stores KV cache chunks in a Valkey (or Redis) server and
supports both **standalone** (``GlideClient``) and **Valkey-Cluster**
(``GlideClusterClient``) topologies, selected via ``cluster_mode``.

I/O is dispatched through an internal ``ValkeyWorkerPool`` of worker
threads; each worker holds its own glide client so a batch of keys
parallelizes across threads and (in cluster mode) across primaries.

Why the ``valkey`` adapter (vs. the RESP adapter)
-------------------------------------------------

The :doc:`RESP <resp>` adapter targets a single Redis/Valkey server
over the native C++ RESP connector. The ``valkey`` adapter is a
first-class Valkey backend: it speaks ``valkey-glide-sync`` directly,
adds Valkey-Cluster support (slot routing, MOVED/ASK redirects, and
topology discovery are handled inside ``GlideClusterClient``), and
preserves copy-reduced SET/GET on multi-megabyte KV chunks (see the
`Valkey L2 Adapter design doc
<https://github.com/LMCache/LMCache/blob/dev/docs/design/v1/distributed/l2_adapters/valkey.md>`_
for the full rationale, including why the sync -- not async -- glide
client is used).

Prerequisites -- installing the glide sync client
-------------------------------------------------

The ``valkey`` adapter requires the ``valkey-glide-sync`` package
(version ``>= 2.3.0``, the first release with both copy-reduced SET
and buffer GET). It is lazy-imported the first time a Valkey worker
starts:

.. code-block:: bash

    pip install 'valkey-glide-sync>=2.3.0'

If the package is missing when a Valkey worker starts, adapter
construction fails with::

    Valkey support requires the glide_sync module.
    Install: pip install 'valkey-glide-sync>=2.3.0'
    (note: the plain 'valkey-glide' package is async-only)

**Required fields:**

- ``startup_nodes``: A ``"host:port[,host:port...]"`` string of seed
  nodes.

  - In **standalone** mode (``cluster_mode: false``, the default) only
    the first entry is used; extras trigger a warning.
  - In **cluster** mode any reachable seed is enough -- glide auto-
    discovers the full topology.

**Optional fields:**

- ``cluster_mode`` (bool, default ``false``): When ``true`` use
  ``GlideClusterClient``; otherwise use standalone ``GlideClient``.
- ``username`` (str, default ``""``): Optional auth username.
- ``password`` (str, default ``""``): Optional auth password.
- ``key_prefix`` (str, default ``""``): Deployment-level key namespace,
  joined with the standard wire key by ``@``. **Must not contain**
  ``@``. Leave empty for a wire-format identical to the unprefixed
  layout.
- ``num_workers`` (int, default ``8``, > 0): Size of the internal worker
  thread pool. Each worker holds its own glide client, so this is the
  real I/O concurrency knob -- raise it to push throughput on large
  batches / cluster deployments.
- ``tls_enable`` (bool, default ``false``): Enable TLS. Required for
  managed Valkey services such as ElastiCache Serverless.
- ``database_id`` (int, optional): Logical database id for
  **standalone** mode. Ignored (with a warning) when
  ``cluster_mode: true``.
- ``request_timeout`` (float, default ``5.0``, > 0): Per-request timeout,
  in seconds.
- ``connection_timeout`` (float, default ``10.0``, > 0): Initial
  connection timeout, in seconds.
- ``max_capacity_gb`` (float, default ``0``, >= 0): Declared aggregate
  capacity in GB used to drive **LMCache-side** eviction. ``0`` (the
  default and the recommended setting) disables global LMCache
  eviction and defers to the Valkey server's own ``maxmemory`` /
  eviction policy; per-``cache_salt`` quotas configured on the
  coordinator still operate either way. Set ``> 0`` (together with an
  ``eviction`` block) only when a single LMCache writer privately owns
  the cluster -- do **not** combine ``> 0`` with server-side eviction
  or a TTL, both of which drift LMCache's byte accounting.
- ``ttl_seconds`` (int, optional): Opt-in per-key TTL in seconds; must
  be a positive integer when set. Omit (default) to write keys with no
  expiry. Enables server-side time-based expiry, required by any
  ``volatile-*`` eviction policy configured on the Valkey server. Do
  **not** combine with ``max_capacity_gb > 0`` (a warning is logged;
  server-side expiry is not reported back to LMCache's byte accounting).
- ``transport`` (str, default ``"tcp"``): How values move. ``"rdma"``
  moves them by RDMA between the L1 arena and a server running the
  valkey-large-object module; see :ref:`valkey-rdma-transport` below.
- ``rdma_provider`` (str, default ``"efa-direct"``): RDMA only.
  ``"efa-direct"`` opens EFA hardware; ``"tcp"`` opens libfabric's
  software provider, which needs no hardware and offers no performance
  benefit, for exercising the RDMA path on an ordinary machine.
- ``rdma_interface`` (str, optional): RDMA only. The fabric domain to pin
  to on a host with more than one card. Omit to let the provider choose.

**Configuration examples:**

.. code-block:: bash

    # Standalone Valkey / Redis server (server-side eviction)
    --l2-adapter '{"type": "valkey", "startup_nodes": "127.0.0.1:6379"}'

    # Valkey Cluster with TLS + auth (managed service)
    --l2-adapter '{
        "type": "valkey",
        "cluster_mode": true,
        "startup_nodes": "seed1.example.com:6379,seed2.example.com:6379",
        "username": "lmcache",
        "password": "secret",
        "tls_enable": true,
        "num_workers": 16
    }'

    # LMCache-driven eviction on a privately-owned Valkey cluster
    --l2-adapter '{
        "type": "valkey",
        "cluster_mode": true,
        "startup_nodes": "10.0.0.1:6379,10.0.0.2:6379",
        "max_capacity_gb": 512,
        "num_workers": 16
    }'

    # Per-key TTL (server-side expiry, no LMCache-side eviction)
    --l2-adapter '{
        "type": "valkey",
        "startup_nodes": "127.0.0.1:6379",
        "ttl_seconds": 3600
    }'

.. _valkey-rdma-transport:

RDMA transport
--------------

With ``"transport": "rdma"`` the adapter stores chunks in a Valkey server
running the
`valkey-large-object <https://github.com/valkey-io/valkey-large-object>`_
module and moves every chunk by **RDMA** through ``valkey-glide-sync``: the
server writes a chunk straight into the L1 memory that will hold it, or
reads one straight out of the L1 memory it lives in. The TCP transport is
copy-reduced, not copy-free (GET materializes the value inside glide, SET
copies it into the command buffer); RDMA removes both copies. Each worker
registers the whole L1 arena with the fabric once at start-up, and every
store or load names the window the chunk occupies; the server then
transfers it with the module's ``BLOB.SET`` and ``BLOB.GET`` (see the
`RDMA transport design doc
<https://github.com/LMCache/LMCache/blob/dev/docs/design/v1/distributed/l2_adapters/valkey_rdma.md>`_).
A glide region carries one transfer at a time, so ``num_workers`` is the
number of transfers in flight.

Everything else -- topologies, the worker pool, wire keys, ``key_prefix``,
LMCache-side eviction -- works the same way with the same fields. Two
fields differ: ``ttl_seconds`` costs a separate ``EXPIRE`` after each store,
because ``BLOB.SET`` takes no expiry argument, and ``request_timeout`` does
**not** bound a transfer, because a posted write cannot be called off.

**Prerequisites:**

- A Valkey server with the **valkey-large-object module** loaded and a
  fabric provider configured: EFA on AWS, or the module's ``Emulated``
  provider with ``"rdma_provider": "tcp"`` for development without
  hardware.
- ``valkey-glide-sync >= 2.6.0`` **built with RDMA support** (published
  wheels are; a source build needs ``GLIDE_SYNC_RDMA=1``).
- **libfabric** ``>= 2.4`` on the client, loaded by glide at runtime. On
  AWS the EFA installer provides it.
- LMCache's L1 memory descriptor, which the distributed storage manager
  passes to the adapter factory in MP mode.

A missing piece fails adapter construction with a message naming it, for
example::

    valkey-glide-sync has RDMA support but libfabric is not available on
    this machine. Install libfabric (for example the libfabric package,
    or the EFA installer on AWS) so the client can open a fabric.

**The two transports store different key types.** RDMA keys are the
module's own data type, which a plain ``GET`` cannot read or ``SET``
replace. Use one transport per ``key_prefix``; switching the transport of
an existing deployment orphans its keys.

**Configuration examples:**

.. code-block:: bash

    # EFA on AWS
    --l2-adapter '{
        "type": "valkey",
        "transport": "rdma",
        "startup_nodes": "kv.internal:6379",
        "num_workers": 8,
        "ttl_seconds": 3600
    }'

    # Valkey Cluster on EFA, pinned to one card
    --l2-adapter '{
        "type": "valkey",
        "transport": "rdma",
        "cluster_mode": true,
        "startup_nodes": "10.0.0.1:6379,10.0.0.2:6379",
        "rdma_interface": "efa0",
        "num_workers": 16
    }'

    # Development: the module's Emulated provider on loopback
    --l2-adapter '{
        "type": "valkey",
        "transport": "rdma",
        "startup_nodes": "127.0.0.1:6379",
        "rdma_provider": "tcp"
    }'

Starting the server for the last example:

.. code-block:: bash

    valkey-server --port 6379 \
        --loadmodule ./libvalkey_large_object.so \
            operating-mode Dram \
            fabric-provider Emulated \
            fabric-interfaces lo

**RDMA notes:**

- Start the MP server with ``--no-l1-use-lazy`` so the whole L1 arena is
  allocated before the workers register it; registering pages that are
  not yet populated is untested.
- Only chunks in the L1 arena can be transferred. In MP mode every chunk
  is; a buffer outside it fails its key and is logged.
- A stored value larger than the destination chunk is refused by the
  server before anything is written, so a chunk-size mismatch cannot
  overrun the neighbouring L1 memory. A smaller value is a miss, as over
  TCP.
- Each worker registers the whole arena, so ``num_workers`` times the L1
  size **must fit the device's cap on registered memory** (on EFA,
  ``max_mr_size`` from ``ibv_devinfo -v``: 96 GiB on g6, 384 GiB on c8gn).
  Exceeding it fails construction with a message naming the product to
  shrink. Registration is fast: tens of milliseconds per worker for a
  32 GiB arena with transparent huge pages, about half a second with 4 KiB
  pages.
- The ``serde`` wrapper still works but stages each chunk through a second
  L1 buffer and a transform pass, forfeiting the copy-free path. Omit
  ``"serde"`` for zero-copy transfers.

Wire key layout
---------------

Each stored value is keyed by::

    [<key_prefix>@]<model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>[@<cache_salt>]

The ``chunk_hash_hex`` component is a uniform content hash and the wire
key contains no Redis hashtag (``{...}``), so keys distribute evenly
across the 16384 cluster slots -- no manual key sharding is required.
