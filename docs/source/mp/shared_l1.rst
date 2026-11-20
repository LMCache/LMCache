Shared Device-DAX L1
====================

Several MP servers can share one Device-DAX region as an L1: a chunk one server
stores is read by the others through their own mappings of the region, without
a network copy. Sharing is an option of the ordinary ``DEVDAX`` L1. A memory
orchestrator process (``lmcache memory``) owns the region's allocation, key
index and object state; KV bytes never pass through it. Without a ``shared``
object, a ``DEVDAX`` L1 stays private and never contacts an orchestrator.
Since the first revision of #5899, the standalone ``SharedL1Manager`` is gone,
``owns_range`` replaces ``owns_device``, ``resolve_l1_affinity`` replaces
``affinity_tag``, and lookups no longer ask private L1s first.

.. code-block:: text

   store:  ReserveWrite -> D2H copies complete -> publish (cache write-back)
           -> FinishWrite (readable everywhere)
   read:   ReserveRead (lease) -> acquire (cache invalidate) -> H2D copies
           complete -> FinishRead
   failed store: queued copies drain -> AbortWrite (never published)

Setup
-----

Run one orchestrator per region, on any host the servers can reach, then make
the shared L1 the first ``--l1-manager`` of every server: the default write
policy places every store in the first L1 (see :doc:`multi_l1`). A private L1
listed after it receives no engine stores; it is the L1 of L2 adapters, which
cannot bind to the shared region.

.. code-block:: bash

   lmcache memory --region-id pool-a --capacity-gb 512 --alignment 2M \
       --listen 0.0.0.0:7700 --state-dir /var/lib/lmcache/memory/pool-a

   lmcache server --chunk-size 256 --supported-transfer-mode lmcache_driven \
       --l1-manager '{"type":"DEVDAX","tag":"cxl","size_gb":512,"path":"/dev/dax0.0",
                      "shared":{"orchestrator":"10.0.0.5:7700","region_id":"pool-a",
                                "client_id":"host-a:5555"}}'

``--visibility-mode`` is ``software_fenced`` (the default: servers write back
and invalidate CPU caches around every copy, for memory that several hosts map
without hardware coherence; extents must be at least page-aligned) or
``coherent`` (no cache maintenance). ``--state-dir`` holds the startup marker
that keeps a second orchestrator off the region.

``shared`` takes ``orchestrator`` (``host:port``), ``region_id``, a unique and
stable ``client_id``, and an optional ``rpc_timeout_seconds`` (default 5). The
region is mapped at the L1's own ``path`` and ``offset_gb``; ``size_gb`` must
cover its capacity, eviction must be ``noop`` (the default), and GPU
registration of the mapping must succeed. Paths may differ between hosts: the
operator asserts that every server's ``path`` and ``offset_gb`` name the same
physical memory, and a mismatch serves wrong KV data without an error. For
tests, ``DEVDAX`` also maps a regular file that servers share under a
``coherent`` orchestrator; this path is test-only, and there is no shared
``DRAM`` L1.

Limits
------

- Cross-host Device-DAX, ``lmcache_driven`` transfer, equal TP1/2/4/8 peers,
  memory tags ``CXL``/``REMOTE``. CacheBlend, P2P, engine-driven or ``auto``
  transfer, and runtime plugins are rejected at startup.
- Peers must use the same model, serving engine and version, TP size, KV
  dtype, chunk size and hash algorithm. Lookup serves only the prefix that
  every rank and object group (including hybrid state) has stored, and rejects
  a peer object whose shapes or dtypes differ from the reader's.
- One shared L1 per server. L2 adapters cannot bind to it; hot-plug returns 409.
- Objects are never evicted. ``delete`` makes objects unreadable for every
  server; ``clear`` deletes the objects this ``client_id`` wrote. Neither
  frees space: deleted, aborted and expired extents stay consumed, so once the
  region is full stores reserve nothing (``L1_ALLOCATION_FAILED`` events)
  until the region is reset. Leases taken before a delete stay readable until
  released.
- There is no authentication or tenant isolation: every attached server reads
  every object. Attach servers of one trust domain only.

Prefill/decode handoff
----------------------

A decode worker can reuse a chunk once the writer's GPU copy has drained,
the writer has published the bytes, and the orchestrator has committed the
object. A prefill HTTP response can arrive before those steps finish. A
decoder that looks up the prefix in that interval sees a miss and may
recompute it. Routers that require a cache handoff need to wait for the
requested prefix to become readable on the decoder before dispatching it.

Failures and reset
------------------

- **Orchestrator unreachable**: reads miss, stores reserve nothing, ``delete``
  returns ``KEY_IS_LOCKED``, ``clear`` deletes nothing, and status shows
  ``state: unavailable`` until an RPC succeeds.
- **Lost FinishWrite reply**: the store reports failure and the extent is not
  reused; a later writer may find the key committed.
- **Malformed grant, or a peer layout that differs from the reader's**: the
  grant is released and logged, never reported as capacity pressure.
- **Epoch change or** ``RESET_REQUIRED``: status shows ``state: disabled``,
  and the L1 serves nothing until the server restarts after a reset.
- **Server restart**: registering the ``client_id`` again retires the previous
  incarnation's writes and leases. Extents are never reused, so copies it left
  in flight cannot overwrite an object other servers read.

An orchestrator restart loses its state and requires a reset. Reset offline:
stop every server using the region, stop the orchestrator, delete
``<state-dir>/<region-id>.marker``, then start the orchestrator and the servers.

In ``GET /status`` (``storage_manager.l1_managers``), the shared entry counts
this server's write grants and read leases: ``memory_used_bytes`` is its
uncommitted write payload, ``memory_total_bytes`` is the region's capacity, and
the ``shared`` section reports ``state`` and ``disabled_reason``. Every
attached server reports the region's capacity, in ``/status`` and in the
``lmcache_mp_l1_*`` gauges, so do not sum it across servers. Placements in a
shared L1 are not reported as fleet cache events.
