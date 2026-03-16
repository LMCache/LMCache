# SPDX-License-Identifier: Apache-2.0

Device-DAX (/dev/dax)
=====================

Overview
--------

The DAX storage plugin maps a ``/dev/dax`` device with ``mmap(MAP_SHARED)``
and uses the mapped region as a fixed-size arena for persisted KV cache
chunks. Reads copy data back into CPU-backed memory objects.


Configuration
-------------

.. code-block:: yaml

   local_cpu: true
   max_local_cpu_size: 80

   storage_plugins: ["dax"]
   extra_config:
     storage_plugin.dax.module_path: lmcache.v1.storage_backend.plugins.dax_backend
     storage_plugin.dax.class_name: DaxBackend

     dax.device_path: "/dev/dax1.0"
     dax.arena_size_gb: 100


Runtime Requirements
--------------------

- ``extra_config['dax.device_path']`` is required and must point to a readable
  and writable DAX device.
- ``LocalCPUBackend`` must be enabled because DAX reads return CPU-backed
  memory objects.


Validation and Current Limits
-----------------------------

- Tensor parallelism is currently limited to TP=1
  (``metadata.world_size == 1``).
- The backend supports only single-tensor chunk layouts. Multi-tensor put
  requests are rejected.
