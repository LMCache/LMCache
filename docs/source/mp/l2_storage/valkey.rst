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
- ``transport`` (str, default ``"tcp"``): Which server commands store each
  chunk. ``"tcp"`` uses plain ``SET``/``GET``. ``"rdma"`` moves chunks by
  RDMA between the L1 arena and a server running the valkey-large-object
  module; see :ref:`valkey-rdma-transport`. ``"blob-tcp"`` stores the same
  module objects over the ordinary connection; see
  :ref:`valkey-blob-tcp-transport`. :ref:`valkey-choosing-a-transport`
  compares them.
- ``rdma_provider`` (str, default ``"efa-direct"``): RDMA only.
  ``"efa-direct"`` opens EFA hardware; ``"tcp"`` opens libfabric's
  software provider, which needs no hardware and offers no performance
  benefit, for exercising the RDMA path on an ordinary machine. Unrelated
  to the ``transport`` values ``"tcp"`` and ``"blob-tcp"``.
- ``rdma_interface`` (str or list of str, optional): RDMA only. The fabric
  domain, or domains, to open on a host with more than one card, as
  ``fi_info -p efa`` lists them (on EFA a name like ``rdmap71s0-rdm``). With
  a list, the workers are spread across the cards round-robin, so make
  ``num_workers`` a multiple of the list length. Omit to let the provider
  choose, which opens one card for every worker. An unknown name fails
  adapter construction with the host's domain names in the message.

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
reads one straight out of the L1 memory it lives in. The ``tcp`` transport
is copy-reduced, not copy-free (GET materializes the value inside glide, SET
copies it into the command buffer); RDMA removes both copies. Each worker
registers the whole L1 arena with the fabric once at start-up, and every
store or load names the window the chunk occupies; the server then
transfers it with the module's ``BLOB.RDMA_SET`` and ``BLOB.RDMA_GET`` (see the
`RDMA transport design doc
<https://github.com/LMCache/LMCache/blob/dev/docs/design/v1/distributed/l2_adapters/valkey_rdma.md>`_).
A glide region carries one transfer at a time, so ``num_workers`` is the
number of transfers in flight.

Everything else -- topologies, the worker pool, wire keys, ``key_prefix``,
LMCache-side eviction -- works the same way with the same fields. Two
fields differ: ``ttl_seconds`` costs a separate ``EXPIRE`` after each store,
because ``BLOB.RDMA_SET`` takes no expiry argument, and ``request_timeout`` does
**not** bound a transfer, because a posted write cannot be called off.

**Prerequisites:**

- A Valkey server with the **valkey-large-object module** loaded and a
  fabric provider configured: EFA on AWS, or the module's ``Emulated``
  provider with ``"rdma_provider": "tcp"`` for development without
  hardware.
- ``valkey-glide-sync`` **built with RDMA support**: the first release with
  the RDMA API, ``2.6.0``, or a build of ``main`` with ``GLIDE_SYNC_RDMA=1``
  until it ships.
- **libfabric** ``>= 2.4`` on the client, loaded by glide at runtime. On
  AWS the EFA installer provides it.
- LMCache's L1 memory descriptor, which the distributed storage manager
  passes to the adapter factory in MP mode.

A missing piece fails adapter construction with a message naming it, for
example::

    valkey-glide-sync has RDMA support but libfabric is not available on
    this machine. Install libfabric (for example the libfabric package,
    or the EFA installer on AWS) so the client can open a fabric.

**RDMA keys are module objects**, the same type ``blob-tcp`` writes; see
:ref:`valkey-choosing-a-transport` for which transports can share a
``key_prefix``.

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

    # Valkey Cluster on EFA, workers spread across two cards
    --l2-adapter '{
        "type": "valkey",
        "transport": "rdma",
        "cluster_mode": true,
        "startup_nodes": "10.0.0.1:6379,10.0.0.2:6379",
        "rdma_interface": ["rdmap71s0-rdm", "rdmap72s0-rdm"],
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
  overrun the neighbouring L1 memory. A smaller value is a miss, as with
  the other transports.
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

.. _valkey-blob-tcp-transport:

blob-tcp transport
------------------

With ``"transport": "blob-tcp"`` the adapter stores chunks as the same
valkey-large-object module objects the RDMA transport writes, but carries
the value over the ordinary connection with the module's ``BLOB.TCP_SET``
and ``BLOB.TCP_GET``. Hosts without an RDMA fabric, libfabric, or an RDMA
build of glide can therefore share one cache with RDMA hosts under the same
``key_prefix``: a chunk stored over either transport loads over the other.

It is not expected to be faster than ``tcp``: the module copies each stored
value into its own buffers and gathers each loaded one into a single reply.
Choose it to share objects with RDMA hosts, not for speed.

**Prerequisites:**

- The **valkey-large-object module** loaded and enabled on every server
  node. No fabric provider is needed.
- ``valkey-glide-sync >= 2.3.0``, the same package the ``tcp`` transport
  uses. No RDMA build and no libfabric.
- With ACLs on, LMCache's user must be allowed ``BLOB.TCP_SET``,
  ``BLOB.TCP_GET`` and ``BLOB.INFO`` (for example ``+@largeobj``) and the
  key ``__lmcache_blob_tcp_probe__``.

Each worker checks the module with ``BLOB.INFO`` when it connects, so a
server that cannot serve the commands fails adapter construction. In cluster
mode the check reaches only the primary that owns its probe key: a shard
elsewhere without the module is not caught at start-up, and its keys fail at
runtime with a per-key warning. The errors look like::

    transport='blob-tcp' needs the valkey-large-object module, but the
    server does not know BLOB.INFO. Load it with
    --loadmodule <path>/libvalkey_large_object.so.

    transport='blob-tcp' needs the valkey-large-object module enabled, but
    the server reports it disabled. Set largeobj.enabled yes.

**Field notes:**

- ``request_timeout`` bounds each request, value included (unlike RDMA).
  Set it above the chunk size divided by the link bandwidth, plus server
  latency. A store that times out may still land; a later load sees either
  the old state or the whole new value.
- ``ttl_seconds`` costs a separate ``EXPIRE`` after each store, because
  ``BLOB.TCP_SET`` takes no expiry argument. If the store succeeds and the
  ``EXPIRE`` fails, the store is reported failed but the object stays with
  no TTL until the chunk is stored again; the RDMA transport behaves the
  same.
- ``has_buffer_get`` in the status and start-up log describes the glide
  client's plain ``GET``, which ``blob-tcp`` does not use.

**Configuration example:**

.. code-block:: bash

    --l2-adapter '{
        "type": "valkey",
        "transport": "blob-tcp",
        "startup_nodes": "kv.internal:6379",
        "key_prefix": "lmcache-prod",
        "num_workers": 8
    }'

.. _valkey-choosing-a-transport:

Choosing a transport
--------------------

.. list-table::
   :header-rows: 1

   * - Transport
     - Server commands
     - Key type
     - Needs
   * - ``tcp``
     - ``SET``, ``GET``
     - plain string
     - any Valkey or Redis
   * - ``rdma``
     - ``BLOB.RDMA_SET``, ``BLOB.RDMA_GET``
     - module object
     - the module, libfabric, RDMA glide, the L1 memory descriptor
   * - ``blob-tcp``
     - ``BLOB.TCP_SET``, ``BLOB.TCP_GET``
     - module object
     - the module

**One key type per prefix.** ``tcp`` stores plain strings. ``rdma`` and
``blob-tcp`` store module objects and can share a ``key_prefix``. Never use
``tcp`` on the same prefix as either. If you do:

- module stores and loads on a string key fail that key with a warning
  containing ``WRONGTYPE``, and so does a ``tcp`` load of a module object;
- a ``tcp`` store silently replaces a module object;
- lookups report hits on keys the other transport wrote, and the loads that
  follow fail.

**Switching transport on an existing prefix.** From ``tcp`` to a module
transport, each existing key fails until it expires or is deleted; keys
written without ``ttl_seconds`` never expire. From a module transport to
``tcp``, keys are replaced one by one as they are stored again. Use a new
prefix instead.

**Restarts and failover.** Module objects do not survive a server restart
or a failover to a replica: the module neither persists nor replicates
them. Plain ``tcp`` keys survive when the server has RDB or AOF enabled.

Wire key layout
---------------

Each stored value is keyed by::

    [<key_prefix>@]<model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>[@<cache_salt>]

The ``chunk_hash_hex`` component is a uniform content hash and the wire
key contains no Redis hashtag (``{...}``), so keys distribute evenly
across the 16384 cluster slots -- no manual key sharding is required.
