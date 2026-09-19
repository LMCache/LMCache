Shared Device-DAX L1 (experimental)
========================================

Multiple LMCache MP servers can reuse one physical KV copy through a shared
Device-DAX pool. A separate Memory Coordinator owns the keys and offsets.
KV bytes move directly between each MP's GPU and its local mapping of the pool.

.. warning::

   This is a bounded experimental path, not a production cache. It requires
   TP1, a qualified cross-host visibility library, and coordinated shutdown.
   There is no eviction, automatic restart recovery, or extent reuse.
   Mapping the wrong window can overwrite unrelated data. Back up and dedicate
   the target range before starting any worker.

The Memory Coordinator is **not** the :doc:`MP Coordinator <coordinator>`.
The latter tracks fleet events; it does not grant shared-memory allocations.
See the :download:`design document <../../design/v1/shared_devdax_l1.md>` for
the state machine, failure model, and native visibility ABI.

Requirements
------------

* Install the full LMCache package with matching CUDA extensions on every MP.
* Provision one shared physical window. Device names and local offsets may
  differ between hosts. Prove visibility in both directions across the full
  intended window. Do not add aliased devices together as separate capacity.
* Use the same model weights, tensor layout, dtype, TP1, chunk size, and cache-key
  settings on participating workers. Use a layout fingerprint that identifies
  these choices. A matching label alone does not validate the underlying bytes.
* Supply an absolute path to a platform-qualified visibility library. LMCache
  does not install or configure the fabric. A generic flush or no-op library is
  not an acceptable substitute. CUDA registration of the full mapping must work.
* Mount the same high-entropy bearer token as a read-only file in the coordinator
  and every MP. Protect it with file permissions; do not put the token in command
  arguments or logs. Use a trusted network or TLS termination.
* Provide a persistent coordinator state directory. Every replacement must see
  the same startup marker. Do not use a pod-local temporary directory for it.

Start the Memory Coordinator
----------------------------

This example uses a dedicated 24 GiB pool with 2 MiB allocation alignment.
The token file and persistent parent directory must already exist. Run exactly
one coordinator process and one Uvicorn worker for the region.

.. code-block:: bash

   export LMCACHE_MEMORY_COORDINATOR_HOST=0.0.0.0
   export LMCACHE_MEMORY_COORDINATOR_PORT=9400
   export LMCACHE_MEMORY_COORDINATOR_TOKEN_FILE=/run/secrets/shared-l1-token
   export LMCACHE_MEMORY_COORDINATOR_STATE_FILE=/var/lib/lmcache/shared-l1/startup.lock
   export LMCACHE_MEMORY_COORDINATOR_REGION_ID=shared-pool-a
   export LMCACHE_MEMORY_COORDINATOR_CAPACITY_BYTES=25769803776
   export LMCACHE_MEMORY_COORDINATOR_ALIGNMENT_BYTES=2097152
   export LMCACHE_MEMORY_COORDINATOR_LAYOUT_ID=qwen25-7b-bf16-tp1-chunk256-v1
   python -m lmcache.v1.memory_coordinator

All settings use the ``LMCACHE_MEMORY_COORDINATOR_`` prefix. ``HOST``, ``PORT``
and ``ALIGNMENT_BYTES`` have the defaults shown above; the remaining values
are required. Alignment must be a positive power of two. ``STATE_FILE`` must
not exist on the first start and remains after shutdown. ``REGION_ID`` names
the physical window; ``LAYOUT_ID`` identifies compatible model/cache layouts.

Attach each MP server
---------------------

Substitute the qualified host-local device and mapping offset. The device below
is only an example. The six shared-L1 settings are opt-in; leaving
``--l1-coordinator-endpoint`` unset keeps the existing private-L1 behavior.

.. code-block:: bash

   lmcache server \
       --host 127.0.0.1 --port 5555 \
       --instance-id mp-a --max-workers 1 --chunk-size 256 \
       --supported-transfer-mode lmcache_driven \
       --worker-reap-timeout-seconds 0 \
       --l1-size-gb 24 --l1-align-bytes 2097152 \
       --l1-devdax-path /dev/dax2.0 --no-l1-use-lazy --shm-name "" \
       --eviction-policy noop \
       --l1-coordinator-endpoint http://coordinator:9400 \
       --l1-coordinator-token-file /run/secrets/shared-l1-token \
       --l1-shared-region-id shared-pool-a \
       --l1-shared-layout-id qwen25-7b-bf16-tp1-chunk256-v1 \
       --l1-shared-mapping-offset-bytes 0 \
       --l1-shared-visibility-library-path /opt/lmcache/libvisibility.so

``--l1-size-gb`` is converted using 1024 cubed; it must match the coordinator's
byte capacity. Alignment, region ID, and layout ID must also match exactly.
``--l1-shared-mapping-offset-bytes`` is host-local and defaults to zero. It must
meet OS/device mapping alignment and place the entire pool inside the device.

Use distinct instance IDs and non-conflicting ports for multiple MPs on one
host. Select the intended GPU for each MP/model pair. The same GPU's CUDA IPC
handles must be accessible to both processes; follow :doc:`deployment`
for container IPC setup. Do not enable worker reaping for this experiment.

Connect a TP1 vLLM worker using the existing MP connector, not the in-process
``LMCacheConnectorV1``:

.. code-block:: bash

   vllm serve /models/compatible-model \
       --tensor-parallel-size 1 --dtype bfloat16 \
       --no-enable-prefix-caching \
       --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":5555}}'

The model must match the chosen layout fingerprint. Disabling GPU prefix caching
helps distinguish shared-L1 reuse from a GPU-local hit during qualification.
This setup enables shared storage; it does not add a P/D proxy or KV-aware router.

Inspection and qualification
----------------------------

``GET /healthz`` and ``GET /readyz`` are unauthenticated process probes. They do
not prove DAX visibility or worker readiness. All ``/v1/*`` endpoints require
``Authorization: Bearer <token>``. Inspect metadata without exposing the token
in a shell argument:

.. code-block:: python

   from lmcache.v1.memory_coordinator.client import MemoryCoordinatorHttpClient

   client = MemoryCoordinatorHttpClient(
       "http://coordinator:9400", "/run/secrets/shared-l1-token"
   )
   try:
       print(client.status().model_dump_json(indent=2))
   finally:
       client.close()

``/v1/region`` returns the fixed contract. ``/v1/status`` returns that contract,
``used_bytes`` (the allocation high-water mark), and ``object_count`` (pending
plus committed records). Aborted extents still consume bytes. These endpoints
do not copy payloads or return a full key directory.

Before benchmarking, compare a cold request with repeated requests on the same
worker and every remote consumer. Check both output equality and each worker's
external-prefix-cache counters. A successful response or a high hit count alone
does not establish correct reuse. Warm up separately from measurement.

When the optional MP Coordinator is used, the existing
``--coordinator-url`` and ``--coordinator-event-reporting`` flags publish shared
Device-DAX placement and capacity events. They do not replace the Memory
Coordinator or make its allocations recoverable. Use only one shared Device-DAX
pool in this deployment; shared usage accounting is grouped by backend.

Limits and troubleshooting
--------------------------

* Use ``noop`` eviction. Clear/delete operations do not reclaim shared extents.
  An exhausted pool requires the coordinated reset below.
* Pending writes have a fixed experimental 60-second TTL. Replacement is lazy:
  a later reservation uses a fresh extent if its entire batch fits. There is no
  background cleanup guarantee. Expired finish/abort calls fail; old bytes remain
  consumed and ``used_bytes`` never decreases. Committed objects do not expire.
  This is not a hardware-qualified timeout: legitimate slow writes may be
  rejected. Expiry does not revoke GPU mappings or recover a fenced client.
* L2 adapters, hybrid DRAM/DAX, GDS L1, MP P2P, CacheBlend, QStore, engine-driven
  transfer, and trace replay are unsupported and rejected.
* A contract mismatch means identity, capacity, alignment, or layout differs.
  Fix the configuration; never bypass the check to make a mapping attach.
* HTTP 507 means the whole absent-key write batch did not fit. No partial
  allocation is made for that batch.
* A stale epoch or ambiguous POST failure fences the client. Do not retry by
  silently adopting another epoch. Stop and investigate, then reset together.
* A visibility or CUDA registration error has no fallback. Verify the platform
  library, device access, mapping geometry, and CUDA registration support.
* An existing startup marker is a deliberate restart refusal, not a stale file
  to remove during a pod restart. The coordinator cannot recover its old index.

Shutdown and reset
------------------

1. Stop new requests. Drain all inference work and GPU transfers.
2. Stop every model worker and MP that can access the region. Do not let a
   controller restart them while resetting the pool.
3. Stop the Memory Coordinator. Verify that no process on any participating
   host retains a mapping or GPU access, including through aliased devices.
4. Only then remove the specific configured startup marker. Restart one
   coordinator and then the MPs. The index is empty and allocation starts over.

Never delete the marker while old workers are alive. Epochs do not revoke CUDA
mappings. The marker must survive pod replacement, including clean shutdown.
It is a safety latch, not a checkpoint. Restore a saved DAX window only after
all readers and writers have stopped; verify it from every participating host.
