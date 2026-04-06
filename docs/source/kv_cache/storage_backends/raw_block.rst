# SPDX-License-Identifier: Apache-2.0

Raw Block Backend
=================

.. _raw-block-overview:

Overview
--------

The raw block backend is an LMCache storage plugin that stores KV cache chunks
directly on a raw block device or file through the Rust
``lmcache_rust_raw_block_io`` extension.

This backend is intended for device-backed local storage where you want LMCache
to issue raw block reads and writes instead of storing chunk files in a
filesystem directory.


Configuration
-------------

Use the plugin name ``raw_block`` in LMCache YAML for backend selection. Do not
use ``RustRawBlockBackend`` for ``store_location`` or
``retrieve_locations``.

.. code-block:: yaml

   chunk_size: 256
   local_cpu: true
   max_local_cpu_size: 5

   storage_plugins: ["raw_block"]
   store_location: "raw_block"
   retrieve_locations: ["raw_block"]

   extra_config:
     storage_plugin.raw_block.module_path: lmcache.v1.storage_backend.plugins.rust_raw_block_backend
     storage_plugin.raw_block.class_name: RustRawBlockBackend

     rust_raw_block.device_path: "/dev/nvme0n1"
     rust_raw_block.use_odirect: true

If ``rust_raw_block.use_odirect`` is enabled, LMCache can automatically align
the local CPU allocator to ``rust_raw_block.block_align`` when
``rust_raw_block.align_local_cpu_allocator`` is left at its default value.


Runtime Requirements
--------------------

- ``extra_config['rust_raw_block.device_path']`` is required and must point to
  a readable and writable block device or file.
- ``LocalCPUBackend`` must be enabled because the raw block backend reads into
  CPU-backed memory objects.


Build the Extension
-------------------

Build and install the Rust extension before starting LMCache:

.. code-block:: bash

   cd rust/raw_block
   pip install maturin
   maturin develop --release


Validation and Current Limits
-----------------------------

- Linux only.
- Tensor parallelism is currently limited to TP=1
  (``metadata.world_size == 1``).
- The backend requires read-write access to the configured device path.
- ``RustRawBlockBackend`` is the plugin class name, but the LMCache YAML
  backend selector remains ``raw_block``.
