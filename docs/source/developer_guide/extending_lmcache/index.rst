Extending LMCache
=================

LMCache is designed to be extensible, allowing integration of custom functionality without modifying the core. The main extension mechanisms are:

- **Storage Plugin Framework** – integrate new storage backends (custom cache storage modules) via a standardized interface.
- **External Remote Connector Framework** – integrate new remote KV store connectors for external/distributed storage systems. This is still supported but has a planned deprecation by v0.5.0. It is replaced by **Remote Storage Plugin Framework**.
- **Remote Storage Plugin Framework** – integrate new remote storage connectors for remote/distributed KV storage systems.
- **Runtime Plugin Framework** – run custom scripts as separate processes alongside LMCache for added functionality.

.. mermaid::

   flowchart LR
      subgraph "LMCache Process"
         direction TB
         core["LMCache Core Engine (Cache Manager & APIs)"]
         runtimePluginMgr["Runtime Plugin Launcher"]
         backendMgr["Storage Backend Interface"]
         storagePluginMgr["Storage Plugin Launcher"]
      end

      runtimePluginMgr -->|"launch"| plugin1["Custom Plugin Script 1"]
      runtimePluginMgr -->|"launch"| plugin2["Custom Plugin Script 2"]

      backendMgr --> CPUBackend[["In-Memory CPU Backend"]]
      backendMgr --> DiskBackend[["Local Disk Backend"]]
      backendMgr --> NIXLBackend[["NIXL Peer Backend"]]
      backendMgr --> RemoteBackend[["Remote connectors"]]

      RemoteBackend --> RedisConnector[["Redis Connector (built-in)"]]
      RemoteBackend --> InfiniConnector[["InfiniStore Connector (built-in)"]]
      RemoteBackend --> MooncakeConnector[["Mooncake Connector (built-in)"]]
      RemoteBackend --> CustomConnector1[["Custom Remote Storage Connector 1"]]
      RemoteBackend --> CustomConnector2[["Custom Remote Storage Connector 2"]]

      storagePluginMgr --> |"StoragePluginInterface"| strplugin1["Custom Storage Backend 1"]
      storagePluginMgr --> |"StoragePluginInterface"| strplugin2["Custom Storage Backend 2"]

**Storage Plugin Framework**  enable LMCache to interface with new storage or transport systems. Developers can implement the standardized ``StoragePluginInterface`` to create a custom storage backend module. Such backends are loaded by LMCache at runtime (via configuration) in addition to the built-in backends that ship with LMCache. Built-in backends include in-memory CPU caching, local disk storage, NVIDIA NIXL (GPU peer-to-peer), and remote stores like Redis, InfiniStore, Mooncake, etc.

To add a custom storage backend, you need to:

1. **Implement** a Python class inheriting from the LMCache ``StoragePluginInterface``, overriding all required methods.
2. **Install** this backend package in the LMCache environment (so that LMCache can import it).
3. **Configure** LMCache to use it by adding an entry to the `storage_plugins` list and specifying the module path and class name in the configuration’s `extra_config`. For example:

   .. code-block:: yaml

      storage_plugins: ["my_custom_storage"]
      extra_config:
         storage_plugin.my_custom_storage.module_path: my_package.my_storage_module
         storage_plugin.my_custom_storage.class_name: MyCustomStorageClass

Multiple storage backends can be enabled simultaneously. They are initialized during LMCache startup. Note that the order of backends can matter: if multiple backends are used, earlier listed backends have higher priority for cache lookups (i.e. LMCache will check those backends first when retrieving KV entries).

**Remote Storage Plugin Framework** allows LMCache's storage layer to connect to new remote KV storage systems. The LMCache `RemoteBackend` uses connector implementations to communicate with various remote stores. Built-in connectors exist for Redis, InfiniStore, MooncakeStore, S3, and more. The plugin system provides the ability to add custom remote storage connectors through dynamic loading, extending remote storage capabilities without modifying core code.

To add a custom remote storage connector, you need to:

1. **Implement** two classes:

   - A ``ConnectorAdapter`` subclass that handles URL parsing and connector instantiation for your scheme
   - A ``RemoteConnector`` subclass that implements the actual storage operations (get, put, exists, etc.)

2. **Package** as an installable Python module (so that LMCache can import it).
3. **Configure** LMCache to use it by adding an entry to the ``remote_storage_plugins`` list and specifying the module path and class name in the configuration's ``extra_config``. For example:

   .. code-block:: yaml

      remote_storage_plugins: ["mystore"]
      extra_config:
         remote_storage_plugin.mystore.module_path: my_package.my_connector_adapter
         remote_storage_plugin.mystore.class_name: MyStoreConnectorAdapter

   With this configuration, when LMCache sees a ``remote_url`` matching your adapter's scheme (e.g., ``mystore://...``), it will import and use your connector for remote operations.

Multiple remote storage plugins can be enabled simultaneously. By implementing a custom connector and configuring it as above, you can integrate new remote backends (databases, distributed KV stores, etc.) without changing LMCache's core code.

**Runtime Plugin Framework** allows running custom scripts alongside LMCache processes. A runtime plugin is launched as a separate subprocess by LMCache’s runtime plugin launcher. Runtime plugins can target specific LMCache roles – the scheduler (controller), worker processes, or all nodes – depending on their filename. They can be written in Python, Bash, or other scripting languages, and are useful for tasks such as logging and metrics, custom cache management policies, health checks, or integration with external systems.

Key points and usage of the runtime plugin system:

- **Configuration:** You can enable runtime plugins via environment variables and the LMCache config file. Set the `runtime_plugin_locations` in your YAML config to point to directories containing plugin scripts. For example:

  .. code-block:: yaml

     runtime_plugin_locations: ["/path/to/plugins"]
     extra_config:
        # (optional plugin-specific settings)

  At runtime, LMCache will scan these directories for plugin files.

- **Environment Variables:** LMCache provides context to plugins through env vars:
  - `LMCACHE_RUNTIME_PLUGIN_ROLE`: the process role (e.g. `SCHEDULER` or `WORKER`) in which the plugin is running.
  - `LMCACHE_RUNTIME_PLUGIN_CONFIG`: a JSON string for any plugin configuration passed through LMCache.
  - `LMCACHE_RUNTIME_PLUGIN_WORKER_ID`: the ID of the current worker (if running on a worker).
  - `LMCACHE_RUNTIME_PLUGIN_WORKER_COUNT`: total number of worker processes in the LMCache cluster.

- **Naming Conventions:** Runtime plugin filenames determine where they execute:
  - Files prefixed with **`scheduler_`** run only on the scheduler process. *(Example: `scheduler_metrics.py` runs on the scheduler only.)*
  - Files prefixed with **`worker_`** run on worker processes. If a numeric worker ID is included (e.g. `worker_0_health.sh`), the plugin runs only on that specific worker. If no ID is included (e.g. `worker_logcollector.py`), the plugin will run on **all** workers.
  - Files prefixed with **`all_`** (or any file without a role prefix) run on all LMCache processes (both the scheduler and every worker). *(Example: `all_monitor.sh` runs on every LMCache node.)*
  - Role names in filenames are case-insensitive. Ensure worker ID (if specified) is numeric and part of the filename (with underscores separating, e.g. `worker_2_custom.py` has three parts, which targets worker ID 2 specifically).

- **Execution Model:** When LMCache starts up, the `RuntiumePluginLauncher` locates and starts each plugin as a subprocess:
  1. **Interpreter Selection:** The runtime plugin launcher checks the script’s shebang (e.g. `#!...`) to decide which interpreter to use. If no shebang is provided, it falls back on the file extension (``.py`` uses the default Python interpreter, ``.sh`` uses Bash, etc.).
  2. **Output Handling:** Stdout and stderr from the plugin processes are captured by LMCache and logged with a prefix (the plugin name), so plugin output appears in LMCache logs for easy debugging/monitoring.
  3. **Lifecycle:** Runtime plugins are launched when the LMCache process starts (if their naming indicates they should run in that process). They will be automatically terminated when the parent LMCache process exits, ensuring no orphan processes.

- **Best Practices:** When writing runtime plugins, consider the following guidelines to ensure they work smoothly with LMCache:
  - Keep plugins lightweight in terms of resource usage and startup time, so they don’t slow down the LMCache process.
  - Use clear and descriptive filenames to reflect their purpose.
  - Include proper error handling within your plugin script to avoid unhandled exceptions causing issues.
  - Use a shebang line at the top of the script for portability (so the correct interpreter is invoked).
  - Validate any configuration input (from `LMCACHE_RUNTIME_PLUGIN_CONFIG` or elsewhere) before use.
  - If a plugin performs lengthy operations, implement timeouts or periodic logging so you can detect if it hangs, and ensure it does not block LMCache’s normal operation.

Together, these extension points – custom storage backends, remote storage connectors, and runtime plugin scripts – let users tailor LMCache's functionality and integrate with external systems in a modular, maintainable way.

.. toctree::
   :maxdepth: 1
   :caption: Extending LMCache

   runtime_plugins
   storage_plugins
   remote_storage_plugins


