# SPDX-License-Identifier: Apache-2.0

Device-DAX (/dev/dax)
=====================

Overview
--------

The DAX storage plugin maps a ``/dev/dax`` device using ``mmap(MAP_SHARED)``
and uses the mapped region as a fixed-size arena for KV cache chunks.
Typical ``/dev/dax`` devices include persistent memory,
CXL-attached memory, and other byte-addressable memory devices.

Data stored on the DAX device may survive process restarts,
but is not guaranteed to be durable.

KV cache data is stored in the DAX region as part of the backend's storage flow.
Reads copy data back into CPU-backed memory objects.


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
     dax.max_dax_size: 100
     dax.restore_workers: 8
     dax.restore_max_regions: 8
     dax.retrieve_staging_slab_bytes: 268435456


Using The Batched Restore Path
------------------------------

The current DAX optimization is a staged batched restore path for retrieval.
It is enabled automatically whenever the DAX backend is configured. No extra
feature flag is required.

The retrieve flow is:

1. Reserve a batched set of readable DAX chunks.
2. Allocate CPU restore buffers from ``LocalCPUBackend``.
3. Copy DAX data into a backend-owned pinned staging slab in coalesced regions.
4. Copy from the staging slab into the final CPU ``MemoryObj`` outputs.
5. Upload those CPU outputs through the normal GPU connector path.

The store flow is unchanged: KV data is still staged through CPU memory before
being written into the DAX arena.

The new DAX tuning knobs control the batched restore path:

- ``dax.restore_workers``: number of persistent worker threads used to execute
  restore regions in parallel.
- ``dax.restore_max_regions``: maximum number of restore regions in one wave.
  Larger values increase parallelism but also increase slab space requirements.
- ``dax.retrieve_staging_slab_bytes``: total size in bytes of the reusable
  pinned retrieve slab. This must be large enough to hold one full chunk per
  configured restore region.

For a first pass, start with:

- ``dax.restore_workers`` equal to the number of CPU workers you want devoted
  to DAX restores
- ``dax.restore_max_regions`` equal to ``dax.restore_workers``
- ``dax.retrieve_staging_slab_bytes`` at least
  ``dax.restore_max_regions * full_chunk_size``, then scale upward if larger
  batched restores are common

If retrieve throughput is low, increase the slab size first, then increase
worker and region counts together. If CPU pressure is high, reduce
``dax.restore_workers`` and ``dax.restore_max_regions``.


Runtime Requirements
--------------------

- ``extra_config['dax.device_path']`` is required and must point to a readable
  and writable DAX device.
- The process must have read-write access to the DAX device
  (e.g., via appropriate permissions or group membership).
- ``LocalCPUBackend`` must be enabled because DAX reads return CPU-backed
  memory objects.


Validation and Current Limits
-----------------------------

- Tensor parallelism is currently limited to TP=1
  (``metadata.world_size == 1``).
- Only single-tensor chunk layouts are supported. Multi-tensor put
  requests are rejected.
- Batched restore uses a backend-owned retrieve staging slab and persistent
  restore executors. The slab and region count can be tuned with
  ``dax.restore_workers``, ``dax.restore_max_regions``, and
  ``dax.retrieve_staging_slab_bytes``.
- Blocking batched restore preserves positional output semantics, while
  asynchronous batched restore returns only the consecutive hit prefix.
