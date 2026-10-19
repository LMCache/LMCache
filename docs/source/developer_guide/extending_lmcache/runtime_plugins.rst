Runtime Plugins
===============

The LMCache runtime plugin system provides the ability to extend functionality by running custom scripts alongside LMCache processes. Plugins can be written in Python and Bash for now, and are managed by the ``RuntimePluginLauncher`` class.

Key Use Cases
-------------
- Start metric reporters for centralized monitoring
- Implement log reporters for log collection systems
- Report process-level metrics to alerting systems
- Implement health checks and service discovery
- Custom cache management operations

Configuration
-------------

Runtime plugins work in both modes through ``RuntimePluginLauncher``.
The MP HTTP server wraps it with ``MPRuntimePluginLauncher``.

Multiprocess server
~~~~~~~~~~~~~~~~~~~

Launch from the LMCache source root to use the bundled examples:

.. code-block:: bash

   lmcache server --l1-size-gb 10 --eviction-policy LRU \
     --runtime-plugin-locations examples/mp_runtime_plugins/

Plugins receive ``LMCACHE_RUNTIME_PLUGIN_CONFIG`` as aggregated JSON with
``mp_config``, ``storage_manager_config``, ``obs_config``, and, when provided,
``http_config`` sections. ``--runtime-plugin-config`` adds a
``runtime_plugin_extra_config`` section to that JSON. MP has no scheduler or
worker roles: the role is empty, the worker ID is 0, and the worker count is 1.
Filename role and worker filters are disabled in this mode. See
:doc:`/mp/configuration` for the server options.

In-process engine
~~~~~~~~~~~~~~~~~

The deprecated in-process integration passes a single ``LMCacheEngineConfig``
JSON and uses the role and worker filters described below.


Environment Variables:
- ``LMCACHE_RUNTIME_PLUGIN_ROLE``: Process role (e.g., ``SCHEDULER``, ``WORKER``)
- ``LMCACHE_RUNTIME_PLUGIN_CONFIG``: JSON string containing plugin configuration
- ``LMCACHE_RUNTIME_PLUGIN_WORKER_ID``: Current worker ID
- ``LMCACHE_RUNTIME_PLUGIN_WORKER_COUNT``: Total worker count in cluster

Configuration File (``lmcache.yaml``):

.. code-block:: yaml

   runtime_plugin_locations:
     - "/path/to/plugins"

   extra_config:
     custom_setting: value

Runtime Plugin Naming Convention
--------------------------------
For the in-process launcher, plugin filenames determine execution targets:

Role-Specific Plugins:
- Format: ``<ROLE>[_<WORKER_ID>][_<DESCRIPTION>].<EXTENSION>``
- Examples:

  - ``scheduler_foo_plugin.py``: Runs only on ``SCHEDULER``
  - ``worker_0_test.sh``: Runs only on worker ID 0
  - ``all_plugin.sh``: Runs on all workers

Notes:
- Role names are case-insensitive
- Worker ID must be numeric when specified
- To target a specific worker ID, the filename must have at least three parts separated by underscores (e.g., `worker_<ID>_<DESCRIPTION>.ext`). A file named `worker_<DESCRIPTION>.ext` will run on all workers.

Execution Model
---------------
1. **Interpreter Detection**:
   - Uses shebang line (e.g., ``#!/opt/venv/bin/python``)
   - Fallback interpreters:

     - ``.py`` → ``python``
     - ``.sh`` → ``bash``

2. **Output Handling**:
   - Stdout/stderr captured continuously
   - Logged with plugin name prefix

3. **Process Management**:
   - Launched as subprocesses
   - Terminated when parent process exits

Example Runtime Plugins
-----------------------
Python Plugin (``scheduler_foo_plugin.py``):

.. literalinclude:: ../../../../examples/runtime_plugins/scheduler_foo_plugin.py
   :language: python
   :linenos:

Bash Plugin (``all_plugin.sh``):

.. literalinclude:: ../../../../examples/runtime_plugins/all_plugin.sh
   :language: bash
   :linenos:

Best Practices
--------------
1. Keep runtime plugins lightweight and efficient
2. Use descriptive naming conventions
3. Implement graceful error handling
4. Include shebang for portability
5. Validate configuration inputs
6. Add timeout mechanisms for long operations