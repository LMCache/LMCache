.. _mp_p2p_deployment:

P2P Deployment Scenarios
========================

:doc:`p2p` explains how P2P KV cache sharing works and lists its flags. This
page shows the two ways to deploy it and when to use each.

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - Scenario A: symmetric sharing
     - Scenario B: main server with headless peers
   * - Nodes
     - Every node runs an inference engine and an LMCache server.
     - One node runs the inference engine and the main LMCache server. The
       other nodes run only an LMCache server ("headless" peers).
   * - Purpose
     - A prefix computed on one node is reused on the others.
     - The headless peers' memory extends the main server's cache.
   * - Who writes where
     - Each node writes only its own memory and reads its peers'.
     - The main server also writes into the headless peers' memory.
   * - Extra flag
     - None.
     - ``--enable-p2p-write`` on the main server only.

Both scenarios need a coordinator for peer discovery and an RDMA-capable
network for good performance; see the requirements in :doc:`p2p`.

Scenario A: symmetric sharing (read-only)
-----------------------------------------

This is the default P2P behavior. Every node caches the KV of the requests it
served and, on a miss, reads the prefix from the peer that holds it.

.. code-block:: text

   node 1                              node 2
   vLLM ──► LMCache server             vLLM ──► LMCache server
            L1 (own KV)  ◄─── read ───────────  miss: look up peers
            miss: look up peers ─── read ────►  L1 (own KV)

Start a coordinator, then on **every** node an LMCache server and an engine:

.. code-block:: bash

   # once, on a host all nodes can reach
   lmcache coordinator --host <COORDINATOR_IP> --port 9300

   # on each node
   lmcache server \
       --host 0.0.0.0 --port 5555 --http-port 8080 \
       --l1-size-gb 100 --eviction-policy LRU \
       --l1-align-bytes 65536 \
       --coordinator-url http://<COORDINATOR_IP>:9300 \
       --coordinator-advertise-ip <NODE_IP> \
       --p2p-advertise-url <NODE_IP>:8500

   vllm serve <model> --port 8000 \
       --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":5555}}'

Nothing is ever written into another node's memory, so each node stays the
sole owner of its L1. See :doc:`p2p` for the step-by-step version, the
single-node test setup and how to verify that peers see each other.

Scenario B: main server with headless peers
-------------------------------------------

Use this when one inference node needs more KV cache memory than it has, and
other machines on the same RDMA network have spare DRAM. A headless peer is a
plain LMCache server with P2P enabled and **no inference engine**. Nothing
would ever reach its memory on its own, so the main server is started with
``--enable-p2p-write`` and stores objects there.

.. code-block:: text

                         main node                         headless peer
   vLLM ── store ──► LMCache server (--enable-p2p-write)   LMCache server
                     local L1                              L1
                       │ usage below threshold: keep
                       │ usage at/above threshold:
                       ├── reserve buffers ───────────────► reserve in L1
                       ├══ RDMA write ════════════════════► bytes land in L1
                       ├── commit ────────────────────────► objects readable
                       └── delete the local copy

   vLLM ── lookup ─► local L1 hit: serve
                     miss ── lookup + RDMA read ──────────► serve from the peer

How the main server decides where an object lives:

* While the local DRAM L1 is less full than
  ``--p2p-write-l1-usage-threshold`` (default ``0.7``), newly stored objects
  stay local.
* From the threshold on, each newly stored batch of objects is written into
  one peer and its local copy is deleted, which keeps local usage around the
  threshold. With several headless peers, batches rotate over them.
* Objects are never moved after the fact: an object that stayed local stays
  local until normal eviction removes it.
* A later request for an object that lives on a peer is served through the
  normal P2P read path.

Start the coordinator as in scenario A, then:

.. code-block:: bash

   # on each headless peer: an LMCache server with P2P, no engine
   lmcache server \
       --host 0.0.0.0 --port 5555 --http-port 8080 \
       --l1-size-gb 200 --eviction-policy LRU \
       --l1-align-bytes 65536 \
       --coordinator-url http://<COORDINATOR_IP>:9300 \
       --coordinator-advertise-ip <PEER_IP> \
       --p2p-advertise-url <PEER_IP>:8500

   # on the main node: the same, plus --enable-p2p-write
   lmcache server \
       --host 0.0.0.0 --port 5555 --http-port 8080 \
       --l1-size-gb 100 --eviction-policy LRU \
       --l1-align-bytes 65536 \
       --coordinator-url http://<COORDINATOR_IP>:9300 \
       --coordinator-advertise-ip <MAIN_IP> \
       --p2p-advertise-url <MAIN_IP>:8500 \
       --enable-p2p-write

   # on the main node: the engine, pointed at the local LMCache server
   vllm serve <model> --port 8000 \
       --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.port":5555}}'

``--enable-p2p-write`` changes two policies (see :doc:`configuration`), each
only if you left it at its default:

* The store policy (``--l2-store-policy``) becomes ``p2p``, which implements
  the placement rule above. If you chose another store policy yourself, it is
  kept and decides what is written to peers.
* The prefetch policy (``--l2-prefetch-policy``) becomes ``skip_l1``, which
  makes an object that is read back from a peer temporary on the main server:
  it is dropped again once the request has used it. If the main server kept a
  second copy of every object it reads back, those copies would fill its L1,
  push it over its eviction watermark, and evict objects that exist nowhere
  else; the measurements below show the difference. If you chose another
  prefetch policy yourself, it is kept and the server logs a warning.

.. warning::

   **Enable writes on only one side of a pair.** If two servers both write
   into each other, an object one of them moved to the other can be moved
   straight back, and the same KV ends up stored twice. In scenario B only the
   main server has ``--enable-p2p-write``; the headless peers do not.

Sizing and tuning
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - Guidance
   * - ``--p2p-write-l1-usage-threshold`` (main)
     - Keep it below ``--eviction-trigger-watermark`` (default ``0.8``).
       Otherwise local eviction starts discarding objects before any are
       written to a peer; the server logs a warning at startup in that case.
       The gap between the two is the room for objects being read back from
       peers: it must hold the KV of all requests that are reading from a
       peer at the same time.
   * - ``--l2-prefetch-policy`` (main)
     - Leave it unset, so that ``--enable-p2p-write`` selects ``skip_l1``. With
       a policy that keeps read-back objects in L1, give the main server
       enough room above the threshold for those copies.
   * - ``--l1-size-gb`` (headless peer)
     - This is the extra cache capacity. A peer evicts with its own eviction
       policy when it fills up. One batch of objects must fit into the peer's
       free space at once, so a peer should be much larger than the KV of one
       request.
   * - ``--p2p-store-timeout`` (main)
     - Deadline for one batch, default 30 s. Keep it below the peer's
       ``--l1-write-ttl-seconds`` (default 600 s).
   * - ``--l1-align-bytes``
     - Use the same value on the main server and the peers; ``65536`` or
       larger improves RDMA transfer performance.

Failure behavior
~~~~~~~~~~~~~~~~

Writing to a peer never puts a request at risk: an object leaves the main
server's memory only after the peer confirmed that it holds it.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Situation
     - What happens
   * - Peer is full
     - The batch is rejected and stays in the main server's L1, where normal
       eviction applies. The peer keeps evicting on its own and accepts
       later batches.
   * - Peer is down or unreachable
     - Batches sent to it fail at the store timeout and stay local. Objects
       that lived only on that peer are cache misses and are recomputed. Until
       the peer is dropped from the peer list (a few missed heartbeats), a
       request that looks an object up on it waits about 3 seconds first. The
       peer is picked up again when it returns.
   * - Main server restarts
     - Its own L1 starts empty; the peers keep what they hold and serve it as
       soon as the main server has rediscovered them.
   * - Main server dies while writing
     - The peer reclaims the half-written buffers after its
       ``--l1-write-ttl-seconds``.

Checking that it works
~~~~~~~~~~~~~~~~~~~~~~

On the main server, ``/status`` lists one P2P adapter per peer under
``storage_manager.l2_adapters``. With writes enabled each shows
``"write_enabled": true`` and the number of batches in flight as
``in_flight_stores``:

.. code-block:: bash

   curl -s http://<MAIN_IP>:8080/status | python3 -m json.tool

On a headless peer, the object count and memory usage of its L1
(``storage_manager.l1_manager`` in ``/status``) grow as the main server fills
up, although the peer serves no engine. The Prometheus metric
``lmcache_mp_l1_staging_bytes`` on a peer shows buffers that are being written
right now; it returns to zero between batches.

Measured on two nodes
---------------------

Setup: one main node (vLLM 0.32 serving ``Qwen/Qwen3-8B`` on one H200 GPU,
plus the main LMCache server) and one headless peer on a second machine,
connected by 8 RoCE links (NIXL 1.5.2 with UCX multi-rail). Prompts are sent
one at a time; vLLM's own prefix cache is off, so every cached token comes
from LMCache. Time to first token (TTFT) is the latency of a one-token
response.

**Cache capacity.** The main server has a 4 GiB L1 and the peer 8 GiB. Fourteen
different prompts of about 3,500 tokens (0.46 GiB of KV each, 6.4 GiB in total)
are sent once, then all fourteen are sent again twice:

.. list-table::
   :header-rows: 1
   :widths: 34 14 14 19 19

   * - Main server configuration
     - Prompts whose KV stayed on the main server after the first pass
     - Prompts whose KV went to the peer
     - Prompt tokens served from cache, first replay
     - Prompt tokens served from cache, second replay
   * - No P2P writes (the peer is unused)
     - at most 11, evicted as the L1 fills
     - 0
     - 9 % (no prompt fully cached)
     - not measured
   * - ``--enable-p2p-write``, threshold 0.7, read-back objects kept in L1
       (``--l2-prefetch-policy retain``)
     - 6
     - 8
     - 74 % (11 of 14 prompts fully cached)
     - 63 % (9 of 14 prompts fully cached)
   * - ``--enable-p2p-write``, threshold 0.5 (read-back objects temporary,
       the ``skip_l1`` prefetch policy that the flag selects)
     - 4
     - 10
     - 95 % (14 of 14 prompts fully cached)
     - 95 % (14 of 14 prompts fully cached)

A fully cached prompt shows 95 % because the last, incomplete chunk of a
prompt is never cached. In the third configuration nothing moved between the
passes: the main server held 52 objects and the peer 131 throughout, with no
object stored twice. In the second configuration the copies kept on the main
server filled its L1 to over 90 %, so some read-backs found no room and four
prompts that had been cached only on the main server were evicted.

**Time to first token.** Median over the prompts of each kind:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Where the prompt's KV is
     - TTFT, prompts of about 3,500 tokens (0.46 GiB of KV)
     - TTFT, prompts of about 14,000 tokens (1.9 GiB of KV)
   * - Not cached: the engine computes it
     - 103 ms
     - 445 ms
   * - In the main server's own L1
     - 50 ms
     - 94 ms
   * - On the headless peer, read back over RDMA
     - 62 ms
     - 118 ms

Reading a prompt's KV back from the peer costs 12 to 24 ms more than a local
hit, and the gain over recomputing grows with the prompt length. Each batch
was written to the peer at 65 to 76 GB/s and read back at 130 to 153 GB/s
(means reported by ``lmcache_mp_l2_store_throughput_GB_per_second`` and
``lmcache_mp_l2_load_throughput_GB_per_second`` for the ``p2p`` adapter).

**Failures**, on the third configuration:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Event
     - Observed
   * - Main server and engine restarted
     - The peer kept its 131 objects. All 10 prompts stored on the peer were
       served from it again; the 4 that had lived on the main server were
       recomputed.
   * - Peer killed
     - Prompts stored on the main server were unaffected (about 40 ms). The
       first two requests for prompts stored on the peer took 3.1 s each;
       after the peer was dropped from the peer list, such requests were
       recomputed at the uncached speed (about 100 ms). No request failed and
       every answer matched the first run.
   * - Peer started again
     - It was rediscovered within seconds; six new prompts were all written to
       it and served from it on replay.

