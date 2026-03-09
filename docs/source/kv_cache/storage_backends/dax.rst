Device-DAX (/dev/dax)
=====================

Overview
--------

The DAX storage plugin maps a ``/dev/dax`` device with ``mmap(MAP_SHARED)``
and uses the mapped region as a fixed-size arena for KV cache chunks.

This backend supports two modes:

- ``tiered``: ``LocalCPUBackend`` remains the allocator and hot tier. DAX keeps
  stored chunks and reads copy data back into a CPU allocation.
- ``primary``: DAX is both the allocator and the storage backend. Reads return
  DAX-backed memory objects directly.


Configuration
-------------

Tiered mode example:

.. code-block:: yaml

   local_cpu: true
   max_local_cpu_size: 80

   storage_plugins: ["dax"]
   extra_config:
     storage_plugin.dax.module_path: lmcache.v1.storage_backend.plugins.dax_backend
     storage_plugin.dax.class_name: DaxBackend

     dax.mode: "tiered"
     dax.device_path: "/dev/dax1.0"
     dax.arena_size_gb: 100

Primary mode example:

In primary mode, DAX replaces the DRAM tier entirely.
``local_cpu`` and ``max_local_cpu_size`` must be ``false`` and ``0`` respectively.

.. code-block:: yaml

   local_cpu: false
   max_local_cpu_size: 0

   storage_plugins: ["dax"]
   extra_config:
     storage_plugin.dax.module_path: lmcache.v1.storage_backend.plugins.dax_backend
     storage_plugin.dax.class_name: DaxBackend

     dax.mode: "primary"
     dax.device_path: "/dev/dax1.0"
     dax.arena_size_gb: 100


Runtime Requirements
--------------------

- ``extra_config['dax.device_path']`` is required and must point to a readable
  and writable DAX device.
- ``tiered`` mode requires ``LocalCPUBackend``.
- ``primary`` mode requires ``local_cpu: false``,
  ``max_local_cpu_size: 0``, and a CUDA worker destination device.


Validation and Current Limits
-----------------------------

- Tensor parallelism is currently limited to TP=1
  (``metadata.world_size == 1``).
- The backend supports only single-tensor chunk layouts. Multi-tensor put and
  allocation requests are rejected.


Troubleshooting
---------------

- ``dax.mode='primary' conflicts with local_cpu=True or max_local_cpu_size > 0``:
  disable the CPU tier for primary mode.
- ``dax.mode='primary' requires a CUDA dst_device``:
  run primary mode on a CUDA worker. CPU-only workers are not supported.
