Valkey RDMA
===========

An L2 adapter that stores KV cache chunks in a Valkey server running the
`valkey-large-object <https://github.com/valkey-io/valkey-large-object>`_
module and moves every chunk by **RDMA**, using the RDMA API of the
official ``valkey-glide`` **sync** Python client. The server writes a chunk
straight into the L1 memory that will hold it, or reads one straight out
of the L1 memory it lives in; the value never travels in the RESP stream.

It is the :doc:`Valkey <valkey>` adapter with its transport swapped.
Standalone and cluster topologies, the worker pool, the wire key layout,
``key_prefix``, ``ttl_seconds``, and LMCache-side eviction all work the
same way and take the same configuration fields.

Why the ``valkey_rdma`` adapter (vs. the ``valkey`` adapter)
-------------------------------------------------------------

The :doc:`valkey <valkey>` adapter is copy-reduced, not copy-free: on GET
the value is materialized inside glide before it is copied into the L1
buffer, and on SET the payload is copied into the command buffer. The
``valkey_rdma`` adapter removes both copies. Each worker thread registers
the whole L1 arena with the fabric once, at start-up, and every store or
load names the window of that registration the chunk occupies; the server
then performs the transfer itself with the module's ``BLOB.SET`` and
``BLOB.GET`` commands (see the
`Valkey RDMA L2 Adapter design doc
<https://github.com/LMCache/LMCache/blob/dev/docs/design/v1/distributed/l2_adapters/valkey_rdma.md>`_
for the full rationale).

Because a glide region carries one transfer at a time, ``num_workers`` is
the number of transfers in flight.

Prerequisites -- server, client, and fabric
-------------------------------------------

- A Valkey server with the **valkey-large-object module** loaded and a fabric
  provider configured. On AWS that is EFA hardware on both the client and
  the server. For development and tests, the module's ``Emulated``
  provider and the adapter's ``"tcp"`` provider carry the same protocol
  over ordinary TCP with no hardware.
- ``valkey-glide-sync`` (version ``>= 2.6.0``) **built with RDMA
  support**. Published wheels include it; a build from source needs
  ``GLIDE_SYNC_RDMA=1``. It is lazy-imported the first time a worker
  starts:

  .. code-block:: bash

      pip install 'valkey-glide-sync>=2.6.0'

- **libfabric** on the client machine (``>= 2.4``), which glide loads at
  runtime. On AWS it comes with the EFA installer.

If any of these is missing when a worker starts, adapter construction
fails naming the missing piece, for example::

    valkey-glide-sync has RDMA support but libfabric is not available on
    this machine. Install libfabric (for example the libfabric package,
    or the EFA installer on AWS) so the client can open a fabric.

The keys this adapter writes are the module's own data type. A plain
``GET`` cannot read them and a plain ``SET`` cannot replace them, so a
server holds either this adapter's keys or the ``valkey`` adapter's keys
under a given ``key_prefix``, not both.

**Required fields:**

- ``startup_nodes``: A ``"host:port[,host:port...]"`` string of seed
  nodes, as for the :doc:`Valkey <valkey>` adapter.

**Optional fields:**

Every optional field of the :doc:`Valkey <valkey>` adapter, plus:

- ``rdma_provider`` (str, default ``"efa-direct"``): ``"efa-direct"``
  opens EFA hardware. ``"tcp"`` opens libfabric's software provider,
  which needs no hardware and offers no performance benefit; it exists so
  the RDMA path can be exercised on an ordinary machine.
- ``rdma_interface`` (str, optional): The fabric domain to pin to on a
  host with more than one card. Omit to let the provider choose.

Two inherited fields mean something slightly different here:

- ``ttl_seconds``: Applied with a separate ``EXPIRE`` after each store,
  because the module's ``BLOB.SET`` takes no expiry argument.
- ``request_timeout``: Does **not** bound a transfer. glide applies no
  timeout to an RDMA transfer, because the server's write cannot be called
  off once it is posted. It still bounds every other command.

**Configuration examples:**

.. code-block:: bash

    # EFA on AWS
    --l2-adapter '{
        "type": "valkey_rdma",
        "startup_nodes": "kv.internal:6379",
        "num_workers": 8,
        "ttl_seconds": 3600
    }'

    # Valkey Cluster on EFA, pinned to one card
    --l2-adapter '{
        "type": "valkey_rdma",
        "cluster_mode": true,
        "startup_nodes": "10.0.0.1:6379,10.0.0.2:6379",
        "rdma_interface": "efa0",
        "num_workers": 16
    }'

    # Development: the module's Emulated provider on loopback
    --l2-adapter '{
        "type": "valkey_rdma",
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

- The adapter needs LMCache's L1 memory descriptor, which the distributed
  storage manager passes to the adapter factory automatically in MP mode.
  If the descriptor is missing or invalid, adapter creation fails with
  ``ValueError`` instead of falling back to a non-RDMA path.
- Start the server with ``--no-l1-use-lazy`` so the whole L1 arena is
  allocated before the workers register it. With lazy allocation the arena
  is only reserved, and registering pages that are not yet populated is
  untested.
- Only chunks that live in the L1 arena can be transferred. In MP mode
  every chunk does; a buffer outside it fails its key and is logged.
- A stored value larger than the destination chunk is refused by the
  server before anything is written, so a chunk-size mismatch can never
  overrun the neighbouring L1 memory. A smaller value is reported as a
  miss, as with the ``valkey`` adapter.
- Each worker registers the whole L1 arena once at start-up, so the
  fabric device holds ``num_workers`` copies of it, and **their product
  must fit the device's cap on registered memory**. On EFA that cap is
  the device's ``max_mr_size`` (``ibv_devinfo -v``): 96 GiB on g6
  instances, 384 GiB on c8gn. Exceeding it fails adapter construction
  with a message naming the product to shrink; lower ``num_workers`` or
  ``--l1-size-gb``. The registrations themselves are fast: tens of
  milliseconds per worker for a 32 GiB arena with transparent huge pages,
  about half a second with 4 KiB pages.
- The ``serde`` wrapper still works, but it stages each chunk through a
  second L1 buffer and a serialize or deserialize pass, which forfeits the
  copy-free path. Omit ``"serde"`` for zero-copy transfers.
