Extending LMCache
=================

LMCache is designed to be extensible, allowing integration of custom functionality without modifying the core. The main extension mechanisms are:

- **Storage Plugin Framework** – integrate new storage backends (custom cache storage modules) via a standardized interface.
- **External Remote Connector Framework** – integrate new remote KV store connectors for external/distributed storage systems.
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
      RemoteBackend --> CustomConnector[["External Connector"]]

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

**External Remote Connectors** (middle section of diagram) allow LMCache’s remote storage layer to connect to new external KV storage systems. The LMCache `RemoteBackend` uses connector implementations to communicate with various external stores. For example, built-in connectors exist for Redis, InfiniStore, and MooncakeStore. To extend LMCache with a new remote connector, you should:

1. **Implement** a class following LMCache’s remote connector interface (similar to existing connectors). This typically involves subclassing the `RemoteConnector` base and providing methods to connect, put, get, etc., for your storage system.
2. **Expose** your connector for dynamic loading. Each remote connector is associated with a URI scheme in the `remote_url`. For instance, if you create a connector for a new storage called “FooStore” with scheme `foo://`, you will want LMCache to use your class whenever a remote URL starts with `foo://`.
3. **Configure** LMCache to recognize and load the connector. In recent versions, LMCache supports dynamic loading of external connectors via configuration. For example, you might include in `extra_config`:

   .. code-block:: yaml

      # Enable custom remote connector for scheme "foo"
      external_connector.foo.module_path: my_package.my_foostore_connector
      external_connector.foo.class_name: FooStoreConnector

   With this configuration, when LMCache sees a `remote_url` like `foo://...`, it will import and use the `FooStoreConnector` class for remote operations.

By implementing a custom connector and configuring it as above, you can integrate new remote backends (databases, distributed KV stores, etc.) without changing LMCache’s core code. The `RemoteConnector` interface typically handles connection setup, data serialization, read/write operations, and error handling for the external store.

**Runtime Plugin Framework** (top section of diagram) allows running custom scripts alongside LMCache processes. A runtime plugin is launched as a separate subprocess by LMCache’s runtime plugin launcher. Runtime plugins can target specific LMCache roles – the scheduler (controller), worker processes, or all nodes – depending on their filename. They can be written in Python, Bash, or other scripting languages, and are useful for tasks such as logging and metrics, custom cache management policies, health checks, or integration with external systems.

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

Together, these extension points – custom storage backends, remote connectors, and runtime plugin scripts – let users tailor LMCache's functionality and integrate with external systems in a modular, maintainable way.

.. toctree::
   :maxdepth: 1
   :caption: Extending LMCache

   runtime_plugins
   storage_plugins


