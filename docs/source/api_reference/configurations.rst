Configuring LMCache
===================

LMCache supports two types of configurations:

1. **Configuration file**: a YAML file that contains the configuration items.
2. **Environment variables**: environment variables that start with ``LMCACHE_``.

To use a configuration file, you can set the ``LMCACHE_CONFIG_FILE`` environment variable to the path of the configuration file.

.. note::

    The environment variable configurations will be ignored if the configuration file is present.


General Configurations
----------------------

Basic cache settings that control the core functionality of LMCache.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - chunk_size
     - LMCACHE_CHUNK_SIZE
     - Size of cache chunks. Default: 256
   * - local_cpu
     - LMCACHE_LOCAL_CPU
     - Whether to enable CPU caching. Values: true/false. Default: true
   * - max_local_cpu_size
     - LMCACHE_MAX_LOCAL_CPU_SIZE
     - Maximum CPU cache size in GB. Default: 5.0
   * - local_disk
     - LMCACHE_LOCAL_DISK
     - Path to local disk cache. Format: "file:///path/to/cache" or null
   * - max_local_disk_size
     - LMCACHE_MAX_LOCAL_DISK_SIZE
     - Maximum disk cache size in GB. Default: 0
   * - remote_url
     - LMCACHE_REMOTE_URL
     - Remote storage URL. Format: "protocol://host:port" or null
   * - remote_serde
     - LMCACHE_REMOTE_SERDE
     - Serialization format. Values: "naive" or "cachegen". Default: "naive"
   * - save_decode_cache
     - LMCACHE_SAVE_DECODE_CACHE
     - Whether to store decode KV cache. Values: true/false. Default: false
   * - error_handling
     - LMCACHE_ERROR_HANDLING
     - Whether to enable error handling. Values: true/false. Default: false

Cache Blending Configurations
-----------------------------

Settings related to cache blending functionality.

.. note::

    Cache blending is not supported in the latest version. We are working on it and will add it back soon.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_blending
     - LMCACHE_ENABLE_BLENDING
     - Whether to enable blending. Values: true/false. Default: false
   * - blend_recompute_ratio
     - LMCACHE_BLEND_RECOMPUTE_RATIO
     - Ratio of blending recompute. Default: 0.15
   * - blend_min_tokens
     - LMCACHE_BLEND_MIN_TOKENS
     - Minimum number of tokens for blending. Default: 256
   * - blend_special_str
     - LMCACHE_BLEND_SPECIAL_STR
     - Separator string for blending. Default: " # # "

Peer-to-Peer Sharing Configurations
-----------------------------------

Settings for enabling and configuring peer-to-peer CPU KV cache sharing and global KV cache lookup.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_p2p
     - LMCACHE_ENABLE_P2P
     - Whether to enable peer-to-peer sharing. Values: true/false. Default: false
   * - lookup_url
     - LMCACHE_LOOKUP_URL
     - URL of the lookup server. Required if enable_p2p is true
   * - distributed_url
     - LMCACHE_DISTRIBUTED_URL
     - URL of the distributed server. Required if enable_p2p is true

Controller Configurations
-------------------------

Settings for the KV cache controller functionality.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_controller
     - LMCACHE_ENABLE_CONTROLLER
     - Whether to enable controller. Values: true/false. Default: false
   * - lmcache_instance_id
     - LMCACHE_LMCACHE_INSTANCE_ID
     - ID of the LMCache instance. Default: "lmcache_default_instance"
   * - controller_url
     - LMCACHE_CONTROLLER_URL
     - URL of the controller server
   * - lmcache_worker_port
     - LMCACHE_LMCACHE_WORKER_PORT
     - Port number for LMCache worker

Nixl (Disaggregated Prefill) Configurations
-------------------------------------------

Settings for Nixl-based disaggregated prefill functionality.

.. note::

    When Nixl is enabled, the following restrictions apply:
    
    - local_cpu must be false
    - max_local_cpu_size must be 0
    - local_disk must be null
    - remote_url must be null
    - save_decode_cache must be false
    - enable_p2p must be false

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_nixl
     - LMCACHE_ENABLE_NIXL
     - Whether to enable Nixl. Values: true/false. Default: false
   * - nixl_role
     - LMCACHE_NIXL_ROLE
     - Nixl role. Values: "sender" or "receiver"
   * - nixl_receiver_host
     - LMCACHE_NIXL_RECEIVER_HOST
     - Host of the Nixl receiver
   * - nixl_receiver_port
     - LMCACHE_NIXL_RECEIVER_PORT
     - Base port of the Nixl receiver
   * - nixl_buffer_size
     - LMCACHE_NIXL_BUFFER_SIZE
     - Transport buffer size for Nixl in bytes
   * - nixl_buffer_device
     - LMCACHE_NIXL_BUFFER_DEVICE
     - Device that Nixl uses
   * - nixl_enable_gc
     - LMCACHE_NIXL_ENABLE_GC
     - Whether to enable Nixl garbage collection. Values: true/false. Default: false




