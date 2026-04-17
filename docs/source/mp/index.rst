Multiprocess Mode
=================

LMCache multiprocess (MP) mode runs LMCache as a **standalone service** that
vLLM instances connect to over ZMQ.  One LMCache server per node can serve
multiple vLLM pods, providing process isolation, shared caching, and
independent resource scaling.

Key Benefits
------------

- **Process isolation** -- LMCache and vLLM run in separate processes (or
  containers), so a cache-related issue does not crash the inference engine.
- **No GIL contention or Python overhead on the inference path** -- By running
  LMCache in a separate process, its Python GIL and CPU work (hashing,
  memory management, L2 I/O) do not compete with vLLM's inference threads.
- **Shared caching across pods** -- Multiple vLLM instances on the same node
  share a single L1 cache, maximizing KV reuse.
- **Independent resource scaling** -- Allocate CPU memory for caching
  independently of GPU memory for inference.
- **Multi-tier storage (L1 + L2)** -- In-memory L1 cache backed by persistent
  L2 storage via NIXL (GDS, POSIX, HF3FS, and more).
- **Built-in observability** -- Prometheus metrics and a telemetry event system
  out of the box.

Prerequisites
-------------

- **vLLM** latest version is recommended for best compatibility
- **LMCache** latest dev branch

Server Variants
---------------

LMCache ships three server entry points:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Entry Point
     - Description
   * - ``lmcache server``
     - **Recommended.** ZMQ + FastAPI HTTP frontend (adds ``/api/healthcheck``
       for K8s probes, ``/api/clear-cache``, ``/api/status``).
       Use ``--engine-type blend`` to enable BlendEngineV2 for cross-request
       KV reuse.
   * - ``python3 -m lmcache.v1.multiprocess.server``
     - (Legacy) ZMQ-only server using MPCacheEngine (no HTTP endpoints).
       Prefer ``lmcache server``.
   * - ``python3 -m lmcache.v1.multiprocess.blend_server_v2``
     - (Legacy) CacheBlend-enabled server. Prefer ``lmcache server --engine-type blend``.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   configuration
   l2_storage
   deployment
   operator
   observability
   architecture
