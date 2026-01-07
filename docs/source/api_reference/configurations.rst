Configuring LMCache
===================

LMCache supports two types of configurations:

1. **Configuration file**: a YAML (recommended) or JSON file that contains the configuration items.
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
     - Path to local disk cache. Format: "file:///path/to/cache".
   * - max_local_disk_size
     - LMCACHE_MAX_LOCAL_DISK_SIZE
     - Maximum disk cache size in GB. Default: 0.0
   * - remote_url
     - LMCACHE_REMOTE_URL
     - Remote storage URL. Format: "protocol://host:port".
   * - remote_serde
     - LMCACHE_REMOTE_SERDE
     - Serialization format. Values: "naive" or "cachegen". Default: "naive"
   * - save_decode_cache
     - LMCACHE_SAVE_DECODE_CACHE
     - Whether to store decode KV cache. Values: true/false. Default: false
   * - use_layerwise
     - LMCACHE_USE_LAYERWISE
     - Whether to enable layerwise pipelining. Values: true/false. Default: false
   * - pre_caching_hash_algorithm
     - LMCACHE_PRE_CACHING_HASH_ALGORITHM
     - Hash algorithm for prefix-caching. Default: "builtin"
   * - save_unfull_chunk
     - LMCACHE_SAVE_UNFULL_CHUNK
     - Whether to save unfull chunks. Values: true/false. Default: false
   * - blocking_timeout_secs
     - LMCACHE_BLOCKING_TIMEOUT_SECS
     - Timeout for blocking operations in seconds. Default: 10
   * - py_enable_gc
     - LMCACHE_PY_ENABLE_GC
     - Whether to enable Python garbage collection. Values: true/false. Default: true
   * - cache_policy
     - LMCACHE_CACHE_POLICY
     - Cache eviction policy (e.g. "LRU", "LFU", "FIFO"). Default: "LRU"
   * - numa_mode
     - LMCACHE_NUMA_MODE
     - NUMA-aware memory allocation mode. Values: "auto" (detect from system), "manual" (use extra_config mapping), null (disabled). When enabled, allocates pinned memory on specific NUMA nodes for better GPU-CPU memory bandwidth. Default: null
   * - external_lookup_client
     - LMCACHE_EXTERNAL_LOOKUP_CLIENT
     - External KV lookup service URI (e.g., "mooncakestore://address"). If null, defaults to LMCache's internal lookup client. Default: null
   * - priority_limit
     - LMCACHE_PRIORITY_LIMIT
     - Caches requests only if priority value ≤ limit. (**Not applicable for PD Disaggregation**) Type: int. Default: None
   * - extra_config
     - LMCACHE_EXTRA_CONFIG={"key": value, ...}
     - Additional configuration as JSON dict. For NUMA manual mode, include "gpu_to_numa_mapping": {gpu_id: numa_node, ...}. Default: {}

Lazy Memory Allocator Configurations
------------------------------------

Settings for the lazy memory allocator that enables gradual memory allocation to reduce startup time and initial memory footprint.

.. note::

    The lazy memory allocator is designed for scenarios with large CPU memory configurations. It starts with a small initial allocation and gradually expands as needed, reducing startup wait time and avoiding unnecessary memory consumption when the full capacity is not required.
    
    **Key characteristics:**
    
    - **One-time expansion**: Memory expands until target size is reached, then stops
    - **No shrinking**: Once allocated, memory is never released back to the system
    - **Automatic activation**: Only activates when ``max_local_cpu_size`` exceeds ``lazy_memory_safe_size``

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_lazy_memory_allocator
     - LMCACHE_ENABLE_LAZY_MEMORY_ALLOCATOR
     - Whether to enable lazy memory allocator. Values: true/false. Default: false
   * - lazy_memory_initial_ratio
     - LMCACHE_LAZY_MEMORY_INITIAL_RATIO
     - Initial memory allocation ratio (0.0-1.0). Determines the fraction of max_local_cpu_size to allocate at startup. Default: 0.2 (20%)
   * - lazy_memory_expand_trigger_ratio
     - LMCACHE_LAZY_MEMORY_EXPAND_TRIGGER_RATIO
     - Memory usage ratio (0.0-1.0) that triggers expansion. When used memory exceeds this ratio of current capacity, expansion begins. Default: 0.5 (50%)
   * - lazy_memory_step_ratio
     - LMCACHE_LAZY_MEMORY_STEP_RATIO
     - Memory expansion step ratio (0.0-1.0). Each expansion adds this fraction of max_local_cpu_size. Default: 0.1 (10%)
   * - lazy_memory_safe_size
     - LMCACHE_LAZY_MEMORY_SAFE_SIZE
     - Threshold in GB above which lazy allocator activates. If max_local_cpu_size ≤ this value, lazy allocator is disabled regardless of enable_lazy_memory_allocator setting. Default: 0.0
   * - reserve_local_cpu_size
     - LMCACHE_RESERVE_LOCAL_CPU_SIZE
     - Reserved system memory in GB that should not be allocated by LMCache. Used to prevent out-of-memory conditions. Default: 0.0
     
Cache Blending Configurations
-----------------------------

Settings related to cache blending functionality.

.. note::

    We have an end-to-end `example <https://github.com/LMCache/LMCache/tree/dev/examples/blend_kv_v1>`_.
    We also have more :doc:`detailed documentation <../kv_cache_optimizations/blending>`.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_blending
     - LMCACHE_ENABLE_BLENDING
     - Whether to enable blending. Values: true/false. Default: false
   * - blend_recompute_ratios
     - LMCACHE_BLEND_RECOMPUTE_RATIOS
     - Ratio of blending recompute. Default: 0.15
   * - blend_check_layers
     - LMCACHE_BLEND_CHECK_LAYERS
     - Layers to determine the recomputed tokens. Default: 1
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
   * - p2p_host
     - LMCACHE_P2P_HOST
     - Ip address. Required if enable_p2p is true
   * - peer_init_ports
     - LMCACHE_PEER_INIT_PORTS
     - Ports for p2p peer init. Required if enable_p2p is true
   * - peer_lookup_ports
     - LMCACHE_PEER_lookup_PORTS
     - Ports for p2p peer lookup. Required if enable_p2p is true
   * - transfer_channel
     - LMCACHE_TRANSFER_CHANNEL
     - Such as `nixl`. Required if enable_p2p is true

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

Disaggregated Prefill Configurations
-------------------------------------------

Settings for disaggregated prefill functionality. The latest/default PD is implemented inside of `lmcache/v1/storage_backend/pd_backend.py`.

.. note::

    When PD is enabled, the following restrictions apply (welcome contributions to remove these restrictions):
    
    - remote_url must be null
    - save_decode_cache must be false
    - enable_p2p must be false

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - enable_pd
     - LMCACHE_ENABLE_PD
     - Whether to enable PD. Values: true/false. Default: false
   * - transfer_channel
     - LMCACHE_TRANSFER_CHANNEL
     - Transfer channel used for PD. Values: "nixl". Default: none
   * - pd_role
     - LMCACHE_PD_ROLE
     - PD role. Values: "sender" (prefiller) or "receiver" (decoder).
   * - pd_buffer_size
     - LMCACHE_PD_BUFFER_SIZE
     - Transport buffer size for PD in bytes. Required for both senders and receivers when enable_pd=true
   * - pd_buffer_device
     - LMCACHE_PD_BUFFER_DEVICE
     - Device for PD buffer. Values: "cpu", "cuda". Required for both senders and receivers when enable_pd=true
   * - nixl_backends
     - LMCACHE_NIXL_BACKENDS
     - List of Nixl transport backends. Useful for non-disaggregated use case (see below). UCX default is sufficient for disagg use case. Default: ["UCX"]
   * - pd_peer_host
     - LMCACHE_PD_PEER_HOST
     - Host for peer connections. Required for receivers to bind to
   * - pd_peer_init_port
     - LMCACHE_PD_PEER_INIT_PORT
     - Initialization port for peer connections. Required for receivers to bind to
   * - pd_peer_alloc_port
     - LMCACHE_PD_PEER_ALLOC_PORT
     - Allocation port for peer connections. Required for receivers to bind to
   * - pd_proxy_host
     - LMCACHE_PD_PROXY_HOST
     - Host for proxy server. Required for senders to connect to inform the proxy when transfer to decoder has been completed
   * - pd_proxy_port
     - LMCACHE_PD_PROXY_PORT
     - Port for proxy server. Required for senders to connect to inform the proxy when transfer to decoder has been completed

P2P Backend Configurations
--------------------------

Settings for P2P (peer-to-peer) backend timeout behavior. These configurations are specified through ``extra_config``.

.. code-block:: yaml

    extra_config:
      p2p_socket_recv_timeout_ms: 30000
      p2p_socket_send_timeout_ms: 10000

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Configuration Key
     - Default
     - Description
   * - p2p_socket_recv_timeout_ms
     - 30000
     - Timeout in milliseconds for socket receive operations
   * - p2p_socket_send_timeout_ms
     - 10000
     - Timeout in milliseconds for socket send operations

Nixl (as a storage backend) Configurations
------------------------------------------

Settings for using Nixl as a storage backend instead of disaggregated prefill. This mode requires additional configurations in ``extra_config``.

.. note::

    This is a different mode from disaggregated prefill. When using Nixl as a storage backend, you need to configure it through ``extra_config``.

.. code-block:: yaml

  
    extra_config: 
      # enable_nixl_storage will disable disaggregated prefill mode.
      enable_nixl_storage: true
      nixl_backend: "POSIX"  # Options: "GDS", "GDS_MT", "POSIX", "HF3FS"
      nixl_path: "/path/to/storage/"
      nixl_file_pool_size: 64

.. list-table::
   :header-rows: 1
   :widths: 30 40

   * - Configuration Key
     - Description
   * - enable_nixl_storage
     - Whether to enable Nixl storage backend. Values: true/false
   * - nixl_backend
     - Storage backend type. Options: "GDS", "GDS_MT", "POSIX", "HF3FS"
   * - nixl_path
     - File system path for Nixl storage
   * - nixl_file_pool_size
     - Number of files in the storage pool


Additional Storage Configurations
---------------------------------

Settings for different storage backends and paths.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - gds_path
     - LMCACHE_GDS_PATH
     - Path for GDS backend
   * - cufile_buffer_size
     - LMCACHE_CUFILE_BUFFER_SIZE
     - Buffer size for cuFile operations

Internal API Server Configurations
----------------------------------

Settings for the internal API server that provides management and debugging APIs for LMCache engines. The API server runs on each worker and scheduler, allowing you to inspect and control LMCache behavior at runtime.

.. note::

    The internal API server provides endpoints for:
    
    - **Metrics**: Performance and cache statistics 
    - **Configuration**: Runtime configuration inspection
    - **Metadata**: Engine and model metadata
    - **Threads**: Thread debugging information
    - **Log Level**: Dynamic log level adjustment
    - **Script Execution**: Run custom Python scripts with access to the LMCache engine

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - internal_api_server_enabled
     - LMCACHE_INTERNAL_API_SERVER_ENABLED
     - Whether to enable internal API server. Default: false
   * - internal_api_server_host
     - LMCACHE_INTERNAL_API_SERVER_HOST
     - Host for internal API server to bind to. Default: "0.0.0.0"
   * - internal_api_server_port_start
     - LMCACHE_INTERNAL_API_SERVER_PORT_START
     - Starting port for internal API server. Port assignment: Scheduler = port_start + 0, Worker i = port_start + i + 1. Example: If port_start=6999, then Scheduler=6999, Worker 0=7000, Worker 1=7001, etc. Default: 6999
   * - internal_api_server_include_index_list
     - LMCACHE_INTERNAL_API_SERVER_INCLUDE_INDEX_LIST
     - List of worker/scheduler indices to enable API server on. Use 0 for scheduler, 1 for worker 0, 2 for worker 1, etc. If null, enables on all workers/scheduler. Example: [0, 1] enables only on scheduler and worker 0. Default: null
   * - internal_api_server_socket_path_prefix
     - LMCACHE_INTERNAL_API_SERVER_SOCKET_PATH_PREFIX
     - If specified, use Unix domain sockets instead of TCP ports. Socket paths will be "{prefix}_{port}". Example: "/tmp/lmcache_api_socket" creates "/tmp/lmcache_api_socket_6999", "/tmp/lmcache_api_socket_7000", etc. Default: null

Plugin Configurations
---------------------

Settings for plugin system.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - plugin_locations
     - LMCACHE_PLUGIN_LOCATIONS
     - List of plugin locations. Default: []

Deprecated Configurations
-------------------------

These configurations are deprecated and may be removed in future versions.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - YAML Config Name
     - Environment Variable
     - Description
   * - audit_actual_remote_url
     - LMCACHE_AUDIT_ACTUAL_REMOTE_URL
     - (Deprecated) URL of actual remote LMCache instance for auditing. Use extra_config['audit_actual_remote_url'] instead
     