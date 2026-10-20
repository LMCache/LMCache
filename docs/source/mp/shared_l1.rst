Shared L1
=========

Several MP servers can share one memory region as an L1: a chunk stored by
one server is retrieved by every other server from its own mapping of the
region, without a network copy. A separate process, the memory orchestrator
(``lmcache memory``), owns the region's allocation, key index and object
state; KV bytes never pass through it. Each server maps the region through a
*medium* on its host:

.. list-table::
   :header-rows: 1
   :widths: 18 30 52

   * - L1 type
     - Medium
     - Use
   * - ``DEVDAX``
     - a Device-DAX character device (``path``)
     - non-coherent memory attached to several hosts; use ``software_fenced``
   * - ``DRAM``
     - a regular file (``path``), created or extended on first use
     - several servers on one host over tmpfs (``/dev/shm``) or hugetlbfs;
       ``coherent``

Start the orchestrator
----------------------

One orchestrator per region, on any host the servers can reach:

.. code-block:: bash

   lmcache memory --region-id pool-a --capacity-gb 512 --alignment 2M \
       --listen 0.0.0.0:7700 --state-dir /var/lib/lmcache/memory/pool-a

``--capacity-gb`` (or ``--capacity-bytes``) is what every server maps from
offset zero of its medium; ``--alignment`` is the extent alignment (default
``2M``); ``--visibility-mode`` is ``software_fenced`` (default; servers flush
CPU caches around every copy) or ``coherent`` (no cache maintenance, for a
region one host maps); ``--state-dir`` holds the startup marker that keeps a
second orchestrator off the region; ``--max-batch-entries`` caps one RPC
(default 4096).

``software_fenced`` requires extent alignment of at least one host page so
independent writers cannot share a CPU cache line. The default ``2M`` alignment
meets this requirement.

Configure the servers
---------------------

A shared L1 is a ``--l1-manager`` with a ``shared`` object. List it first:
writes follow ``--l1-manager`` order and only out-of-memory writes move on,
so stores land where every server can read them and a following DRAM L1
takes them once the region is full. Lookups always ask local L1s first.

.. code-block:: bash

   # CXL region shared across hosts, private DRAM as overflow
   lmcache server --chunk-size 256 --eviction-policy LRU \
       --supported-transfer-mode lmcache_driven \
       --l1-manager '{"type":"DEVDAX","tag":"cxl","size_gb":512,"path":"/dev/dax0.0",
                      "shared":{"orchestrator":"10.0.0.5:7700","region_id":"pool-a","client_id":"host-a:5555"}}' \
       --l1-manager '{"type":"DRAM","tag":"_default","size_gb":64}'

   # tmpfs region shared by several servers on one host
   # (orchestrator started with --capacity-gb 64 --visibility-mode coherent)
   lmcache server --chunk-size 256 --supported-transfer-mode lmcache_driven \
       --l1-manager '{"type":"DRAM","tag":"pool","size_gb":64,"path":"/dev/shm/pool-a",
                      "shared":{"orchestrator":"127.0.0.1:7700","region_id":"pool-a","client_id":"host-a:5555"}}'

``shared`` requires ``orchestrator`` (``host:port``), ``region_id`` (matching
``--region-id``), and ``client_id``. Give each MP server a unique, stable client
ID; restarting under the same ID retires its previous incarnation. Set a
different ID on the peer server. ``rpc_timeout_seconds`` defaults to 5.
``size_gb`` must cover the region capacity; shared eviction defaults to and
requires ``noop``. GPU registration must succeed; startup fails rather than
using pageable staging. The operator establishes that every server's ``path``
names the same physical memory.

Limits
------

- ``lmcache_driven`` transfer with equal TP1/2/4/8 peers. CacheBlend, P2P,
  and runtime plugins are rejected.
- One shared L1 per server; L2 adapters cannot use it as ``affinity_tag``;
  GDS L1s cannot be shared.
- Shared objects are never evicted or deleted and extents are never reused,
  so once the region is full new stores overflow to the next L1 until the
  region is reset. Shared placements are not reported as fleet cache events.

Failures and reset
------------------

An unreachable orchestrator makes reads miss and stores overflow to private
L1s. The shared L1 reports unhealthy until it
answers again. A server that dies mid-store leaves its chunks unreadable
(others recompute them) until its write TTL expires or it restarts under the
same ``client_id``. Set the orchestrator TTL with ``--write-ttl-seconds``
(default 600); expired or aborted extents are not reused. An
orchestrator restart loses its in-memory state, so it comes back requiring a
reset and every server's shared L1 stops serving. Reset offline: stop every
server using the region, stop the orchestrator, delete
``<state-dir>/<region-id>.marker``, start the orchestrator, start the servers.

Tensor parallel serving
-----------------------

Run one MP server per host and point all local engine workers at it. Both
hosts must map the same physical region and use the same model revision,
cache model name, serving engine/version, TP size, KV dtype, chunk size and
hash algorithm. Verify that ``cache_context_meta`` in both servers' ``/status``
reports the same ``model_name``. vLLM can resolve a model ID to a local cache
path, so matching Hugging Face IDs alone may not suffice. Use identical model
directory paths on both hosts; their weight files can be symlinks to separate
disks. A served-model alias does not replace the model name in cache keys.
Independent chunk negotiation
on each MP server must converge to the same size before sharing data.

Each TP worker stores its rank's shard. Lookup returns only the prefix available
across every required rank and object group, including hybrid state. A peer
with incompatible tensor shapes or dtypes is rejected. This does not reshard
KV between TP sizes or translate layouts between serving engines.

Status and metrics
------------------

``GET /status`` includes each L1 under ``storage_manager.l1_managers``. The
shared entry reports the region, epoch, client identity, visibility mode, DMA
path, reachability and fencing reason. Its ``region_usage`` reports global
allocated bytes, valid/writing objects and read leases; usage is cached for
up to one second. Allocation includes abandoned extents because they are not
reused. Other L1 entries continue reporting their own private pools.

The existing ``lmcache_mp_l1_memory_usage_bytes``,
``lmcache_mp_l1_write_chunks_total`` and ``lmcache_mp_l1_read_chunks_total``
metrics carry ``l1_tag``. Capacity is grouped by the backing medium in status.
A shared region's global capacity appears on each attached server; summing
those server reports counts the same physical region more than once.
