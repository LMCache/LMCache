
Nixl
====

.. _nixl-overview:

Overview
--------

NIXL (NVIDIA Inference Xfer Library) is a high-performance library designed for accelerating point to point communications in AI inference frameworks. It provides an abstraction over various types of memory (CPU and GPU) and storage through a modular plug-in architecture, enabling efficient data transfer and coordination between different components of the inference pipeline.

LMCache supports using NIXL as a storage backend, allowing using NIXL to save either GPU or CPU memory into storage.

Prerequisites
~~~~~~~~~~~~~

- **LMCache**: Install with ``pip install lmcache``
- **NIXL**: Install from `NIXL GitHub repository <https://github.com/ai-dynamo/nixl>`_
- **Model Access**: Valid Hugging Face token (HF_TOKEN) for Llama 3.1 8B Instruct

Ways to configure LMCache NIXL Offloading
-----------------------------------------

**Configuration File**:

Passed in through ``LMCACHE_CONFIG_FILE=lmcache-config.yaml``

Example ``lmcache-config.yaml`` for POSIX backend:

.. code-block:: yaml

    chunk_size: 256
    nixl_buffer_size: 1073741824 # 1GB
    nixl_buffer_device: cpu
    extra_config:
      enable_nixl_storage: true
      nixl_backend: POSIX
      nixl_pool_size: 64
      nixl_path: /mnt/nixl/cache/
      use_direct_io: True

Key settings:

- ``nixl_buffer_size``: buffer size for NIXL transfers.

- ``nixl_pool_size``: number of descriptors opened at init time for nixl backend. Set to 0 for dynamic mode.

- ``nixl_path``: directory under which the storage files will be saved (e.g. /mnt/nixl/). Needed for NIXL backends that store to file.

- ``nixl_buffer_device``: dictates where the memory managed by NIXL should be on. "cpu" or "cuda" is supported for "GDS" and "GDS_MT" backends - for "POSIX", "HF3FS" & "OBJ", must be "cpu".

- ``nixl_backend``: configuration of which nixl backend to use for storage.

  .. note::

    Supported backends are: ["GDS", "GDS_MT", "POSIX", "HF3FS", "OBJ"].

    Backend specific params should be provided via ``extra_config.nixl_backend_params``. Please refer to NIXL documentation for specifics.

Example ``lmcache-config.yaml`` for OBJ backend using S3 API:

.. code-block:: yaml

    chunk_size: 256
    nixl_buffer_size: 1073741824 # 1GB
    nixl_buffer_device: cpu
    extra_config:
      enable_nixl_storage: true
      nixl_backend: OBJ
      nixl_pool_size: 64
      nixl_path: /mnt/nixl/cache/
      nixl_backend_params:
        access_key: <your_access_key>
        secret_key: <your_secret_key>
        bucket: <your_bucket>
        region: <your_region>

Dynamic Mode
~~~~~~~~~~~~~

Nixl Storage Backend also supports a dynamic mode, which creates nixl storage descriptors on demand instead of at init time.

In order to use dynamic mode, extra_config.nixl_pool_size should be set to 0.

Restrictions
^^^^^^^^^^^^

- Dynamic mode is currently only supported for nixl OBJ backend.
- save_unfull_chunk must be set to False.

Example ``lmcache-config.yaml`` for OBJ backend with dynamic mode:

.. code-block:: yaml

  chunk_size: 256
  local_cpu: False
  save_unfull_chunk: False
  enable_async_loading: False # set to True to test async loading
  # buffer size has to be divisible by chunk size
  # 2880MiB is divisible by 256 token chunk for Qwen3-4B/8B/32B
  nixl_buffer_size: 3019898880
  nixl_buffer_device: cpu
  extra_config:
    enable_nixl_storage: true
    nixl_backend: OBJ
    nixl_pool_size: 0
    nixl_presence_cache: False
    nixl_async_put: False
    nixl_backend_params:
      access_key: <your_access_key>
      secret_key: <your_secret_key>
      bucket: <your_bucket>
      region: <your_region>
      endpoint_override: https://url-to-object-storage
      ca_bundle: path to self-signed certificate # remove this line if not using self-signed certificate
