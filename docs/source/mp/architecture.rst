Architecture & Developer Guide
===============================

This page describes the internal architecture of LMCache multiprocess mode.
It is aimed at developers who want to understand, debug, or extend the system.

.. contents::
   :local:
   :depth: 2

High-Level Architecture
-----------------------

.. code-block:: text

    vLLM Instance(s)
         |
         | ZMQ (tcp)
         v
    MessageQueueServer (mq.py)
         |
         | dispatch by RequestType
         v
    MPCacheEngine (server.py)
         |
         |--- TokenHasher / SessionManager
         |
         v
    StorageManager (distributed/storage_manager.py)
         |
         |--- L1Manager (l1_manager.py)
         |       |--- L1MemoryManager (memory allocator)
         |       |--- TTLLock per object (read/write)
         |
         |--- StoreController  -----> L2 Adapter(s) (async L1->L2 push)
         |--- PrefetchController ---> L2 Adapter(s) (async L2->L1 load)
         |--- EvictionController ----> L1Manager (watermark-triggered eviction)
         |
         v
    PrometheusController + TelemetryController (observability)

Server Variants
---------------

All three server entry points share the same ``MPCacheEngine`` and
``StorageManager`` core.

**``server.py``** -- The default ZMQ-only server.  Creates an ``MPCacheEngine``
and a ``MessageQueueServer``, registers handlers for all core
``RequestType`` values, and blocks in a keep-alive loop.

**``blend_server_v2.py``** -- Extends ``MPCacheEngine`` with ``BlendEngineV2``,
which adds CacheBlend operations (``CB_REGISTER_KV_CACHE``,
``CB_LOOKUP_PRE_COMPUTED``, ``CB_STORE_PRE_COMPUTED``,
``CB_RETRIEVE_PRE_COMPUTED``, ``CB_STORE_FINAL``).  Enables non-prefix KV
cache reuse across document paragraphs.

**``http_server.py``** -- Wraps ``run_cache_server()`` (from ``server.py``)
inside a FastAPI application.  Adds ``/api/healthcheck`` for Kubernetes probes,
``POST /api/clear-cache`` for clearing all KV cache data in L1 (CPU) memory,
and ``/api/status`` for inspecting detailed internal state.
The ZMQ server runs as part of the same process.

ZMQ Protocol
------------

Communication between vLLM and LMCache uses ZMQ (DEALER/ROUTER pattern).

**RequestType enum** (defined in ``protocols/base.py``):

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Request Type
     - Handler Type
     - Description
   * - ``REGISTER_KV_CACHE``
     - SYNC
     - Register GPU KV cache tensors for a vLLM instance.
   * - ``UNREGISTER_KV_CACHE``
     - SYNC
     - Unregister KV cache tensors.
   * - ``STORE``
     - BLOCKING
     - Store KV cache chunks from GPU to L1 (CPU).
   * - ``RETRIEVE``
     - BLOCKING
     - Copy KV cache chunks from L1 (CPU) back to GPU.
   * - ``LOOKUP``
     - BLOCKING
     - Check prefix cache hits and submit prefetch tasks.
   * - ``FREE_LOOKUP_LOCKS``
     - SYNC
     - Release read locks from a cancelled lookup.
   * - ``END_SESSION``
     - SYNC
     - Remove session state for a finished request.
   * - ``CLEAR``
     - SYNC
     - Clear all cached data.
   * - ``GET_CHUNK_SIZE``
     - SYNC
     - Return the server's chunk size.
   * - ``NOOP``
     - SYNC
     - Debug ping -- returns "OK".
   * - ``CB_REGISTER_KV_CACHE``
     - SYNC
     - (Blend) Register CacheBlend KV buffer.
   * - ``CB_UNREGISTER_KV_CACHE``
     - SYNC
     - (Blend) Unregister CacheBlend KV buffer.
   * - ``CB_STORE_PRE_COMPUTED``
     - BLOCKING
     - (Blend) Store pre-computed paragraph chunks.
   * - ``CB_LOOKUP_PRE_COMPUTED``
     - BLOCKING
     - (Blend) Lookup pre-computed paragraph chunks.
   * - ``CB_RETRIEVE_PRE_COMPUTED``
     - BLOCKING
     - (Blend) Retrieve pre-computed paragraph chunks to GPU.
   * - ``CB_STORE_FINAL``
     - BLOCKING
     - (Blend) Store final blended chunks.

**Handler types:**

- **SYNC** -- Runs directly in the ZMQ main loop (fast, non-blocking).
- **BLOCKING** -- Dispatched to a thread pool (may involve GPU copies or I/O).

Config System
-------------

Each config module exposes a composable triple:

.. code-block:: text

    (DataclassConfig, add_*_args(parser), parse_args_to_*_config(args))

``server.py:parse_args()`` composes them:

.. code-block:: python

    parser = argparse.ArgumentParser(...)
    add_mp_server_args(parser)        # from multiprocess/config.py
    add_storage_manager_args(parser)  # from distributed/config.py
      # which internally calls add_l2_adapters_args(parser)
    add_prometheus_args(parser)       # from mp_observability/config.py
    add_telemetry_args(parser)        # from mp_observability/telemetry/config.py

Both ``blend_server_v2.py`` and ``http_server.py`` reuse this pattern, adding
``add_http_frontend_args()`` for the HTTP variant.

Distributed Storage
-------------------

StorageManager
~~~~~~~~~~~~~~

``lmcache/v1/distributed/storage_manager.py``

The top-level manager that wires together L1, L2, and all controllers.  Key
methods:

- ``reserve_write()`` / ``finish_write()`` -- Two-phase write into L1.
- ``submit_prefetch_task()`` / ``query_prefetch_status()`` -- Async lookup +
  L2 prefetch.
- ``read_prefetched_results()`` / ``finish_read_prefetched()`` -- Read
  prefetched data from L1 with automatic lock management.

L1Manager
~~~~~~~~~

``lmcache/v1/distributed/l1_manager.py``

Manages objects in CPU memory with a state machine:

.. code-block:: text

    None --> write_locked --> ready --> read_locked
              (reserve_write)  (finish_write)  (reserve_read)
                                  |                |
                                  v                v
                               evictable      finish_read -> ready

Each object has two ``TTLLock`` instances (read and write) with configurable
timeouts to prevent deadlocks from crashed clients.

The ``L1MemoryManager`` handles the underlying memory allocation (lazy growth
up to ``--l1-size-gb``).

L2 Adapters
~~~~~~~~~~~

``lmcache/v1/distributed/l2_adapters/``

The ``L2AdapterInterface`` (in ``base.py``) defines three async task methods:

- ``submit_store_task(key, data)`` -- Push data to L2.
- ``submit_lookup_and_lock_task(keys)`` -- Check if keys exist in L2.
- ``submit_load_task(keys, layout_desc)`` -- Load data from L2 into L1.

The factory function ``create_l2_adapter()`` (in ``__init__.py``) uses
``isinstance()`` on the config type to instantiate the correct adapter.

New adapter types are registered via ``register_l2_adapter_type()`` in
``config.py``.

Controllers
~~~~~~~~~~~

**StoreController** (``storage_controllers/store_controller.py``):
Event-driven background thread that uses ``select.poll()`` on listener eventfd
and adapter store eventfds.  When new objects appear in L1 (signaled via
``StoreListener``), it submits async store tasks to each L2 adapter based on
the ``StorePolicy``.

**EvictionController** (``storage_controllers/eviction_controller.py``):
Periodically checks L1 memory usage against the watermark threshold.  When
triggered, evicts objects using the configured policy (LRU) until usage drops
below the target.

**PrefetchController** (``storage_controllers/prefetch_controller.py``):
Handles L2 lookup and load requests submitted by ``StorageManager`` during
``LOOKUP`` RPCs.  When keys are not in L1, it queries L2 adapters and loads
found data back into L1.

Request Flows
-------------

LOOKUP Flow
~~~~~~~~~~~

.. code-block:: text

    vLLM                MPCacheEngine          StorageManager         L1Manager       L2 (PrefetchController)
     |                       |                       |                    |                    |
     |---LOOKUP(key)-------->|                       |                    |                    |
     |                       |--submit_prefetch------>|                    |                    |
     |                       |                       |--reserve_read----->|                    |
     |                       |                       |<--hit_count--------|                    |
     |                       |                       |--submit_prefetch_request--------------->|
     |                       |                       |    (remaining keys)                     |
     |                       |--query_prefetch------->|                    |                    |
     |                       |                       |--query_prefetch_result----------------->|
     |                       |<--found_count----------|                    |                    |
     |<--found_count---------|                       |                    |                    |

STORE Flow
~~~~~~~~~~

.. code-block:: text

    vLLM                MPCacheEngine          StorageManager         L1Manager
     |                       |                       |                    |
     |---STORE(key,blocks)-->|                       |                    |
     |                       |--reserve_write-------->|                    |
     |                       |                       |--reserve_write---->|
     |                       |                       |<--memory_objs------|
     |                       |  (GPU->CPU copy)      |                    |
     |                       |--finish_write--------->|                    |
     |                       |                       |--finish_write----->|
     |                       |                       |                    |
     |                       |                       |  [StoreController detects new objects]
     |                       |                       |  [async L1->L2 push via adapters]
     |<--event_handle--------|                       |                    |

RETRIEVE Flow
~~~~~~~~~~~~~

.. code-block:: text

    vLLM                MPCacheEngine          StorageManager         L1Manager
     |                       |                       |                    |
     |---RETRIEVE(key)------>|                       |                    |
     |                       |--read_prefetched------>|                    |
     |                       |                       |--unsafe_read------>|
     |                       |                       |<--memory_objs------|
     |                       |  (CPU->GPU copy)      |                    |
     |                       |--finish_read_prefetch->|                    |
     |                       |                       |--finish_read------>|
     |<--event_handle--------|                       |                    |

Observability Internals
-----------------------

**PrometheusController** is a global singleton (initialized at server startup).
It runs a daemon thread that periodically calls ``log_prometheus()`` on all
registered loggers (``StorageManagerStatsLogger``, ``L1ManagerStatsLogger``).
Each logger atomically snapshots its stats, resets to zero, and pushes values
to Prometheus counters.

**TelemetryController** is also a global singleton.  It maintains an in-memory
event queue (bounded by ``--telemetry-max-queue-size``).  A drain thread reads
events and dispatches them to registered processors (e.g., ``LoggingProcessor``).

How to Extend
-------------

Adding a new L2 adapter
~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a config class subclassing ``L2AdapterConfigBase`` with
   ``from_dict()`` and ``help()`` methods.
2. Call ``register_l2_adapter_type("my_adapter", MyAdapterConfig)`` at module
   level.
3. Create an adapter class implementing ``L2AdapterInterface``.
4. Add an ``isinstance()`` branch in ``create_l2_adapter()``
   (``l2_adapters/__init__.py``).

Adding a telemetry processor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Create a config class subclassing ``TelemetryProcessorConfig`` with
   ``from_dict()`` and ``help()`` methods.
2. Call ``register_telemetry_processor_type("my_proc", MyProcConfig)`` at
   module level.
3. Create a processor class implementing ``TelemetryProcessor``
   (``on_new_event()``, ``shutdown()``).
4. Add a factory branch in the telemetry controller's processor creation code.

Adding a new request type
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Add a new member to ``RequestType`` in ``protocols/base.py``.
2. Create a ``ProtocolDefinition`` in the appropriate ``protocols/*.py`` file
   (engine, controller, or debug).
3. Implement the handler method on ``MPCacheEngine`` (or ``BlendEngine``).
4. Register the handler in ``run_cache_server()`` via ``add_handler_helper()``.

Key Source Files
----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Purpose
   * - ``lmcache/v1/multiprocess/server.py``
     - MPCacheEngine + ZMQ server entry point
   * - ``lmcache/v1/multiprocess/config.py``
     - MPServerConfig, HTTPFrontendConfig
   * - ``lmcache/v1/multiprocess/blend_server_v2.py``
     - BlendEngineV2 (extends MPCacheEngine)
   * - ``lmcache/v1/multiprocess/http_server.py``
     - FastAPI wrapper with health check and many other useful APIs
   * - ``lmcache/v1/multiprocess/protocols/base.py``
     - RequestType, HandlerType, ProtocolDefinition
   * - ``lmcache/v1/distributed/storage_manager.py``
     - StorageManager (top-level manager)
   * - ``lmcache/v1/distributed/config.py``
     - StorageManagerConfig hierarchy
   * - ``lmcache/v1/distributed/l1_manager.py``
     - L1Manager (object state machine)
   * - ``lmcache/v1/distributed/l2_adapters/config.py``
     - L2 adapter config registry
   * - ``lmcache/v1/distributed/l2_adapters/base.py``
     - L2AdapterInterface
   * - ``lmcache/v1/distributed/storage_controllers/store_controller.py``
     - StoreController (event-driven L1->L2)
   * - ``lmcache/v1/distributed/storage_controllers/eviction_controller.py``
     - EvictionController (watermark-triggered)
   * - ``lmcache/v1/distributed/storage_controllers/prefetch_controller.py``
     - PrefetchController (L2->L1 on miss)
   * - ``lmcache/v1/mp_observability/config.py``
     - PrometheusConfig
   * - ``lmcache/v1/mp_observability/prometheus_controller.py``
     - PrometheusController singleton
   * - ``lmcache/v1/mp_observability/telemetry/config.py``
     - TelemetryConfig
   * - ``lmcache/v1/mp_observability/telemetry/controller.py``
     - TelemetryController singleton
