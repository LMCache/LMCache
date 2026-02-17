.. _internal_api_server:

Configuring the Internal API Server
====================================

The ``internal_api_server`` provides APIs for managing the LMCache engine. Below are the configuration options and usage examples.

Configuration Parameters
------------------------

The following parameters can be configured in the YAML file:

.. code-block:: yaml

    # Enable/disable the internal API server
    internal_api_server_enabled: True
    # Base port for the API server
    # actual_port = internal_api_server_port_start + index
    # Scheduler → 6999 + 0 = 6999
    # Worker 0 → 6999 + 1 = 7000
    internal_api_server_port_start: 6999
    # List of scheduler/worker indices: 0 for scheduler, 1 for worker 0, 2 for worker 1, etc.
    internal_api_server_include_index_list: [0, 1]
    # Socket path prefix for the API server. If configured, the server will use a Unix socket instead of listening on a port.
    internal_api_server_socket_path_prefix: "/tmp/lmcache_internal_api_server/socket"

    # Actual socket files will be:
    #   /tmp/lmcache_internal_api_server/socket_6999 (scheduler)
    #   /tmp/lmcache_internal_api_server/socket_7000 (worker 0)

.. _testing_internal_api_server:

Testing the Server
---------------------------------------

You can test the server by querying the relevant endpoints.

`/metrics` endpoint for metrics:

.. code-block:: bash

    curl http://localhost:7000/metrics

`/conf` endpoint for configuration:

.. code-block:: bash

    # Get current configuration
    curl http://localhost:7000/conf

    # Update one or more config values (Experimental)
    curl -X POST http://localhost:7000/conf \
      -H "Content-Type: application/json" \
      -d '{"min_retrieve_tokens": 512, "save_decode_cache": true}'

.. warning::

    The ``POST /conf`` feature is currently **experimental**. At present,
    all configuration keys are mutable at runtime by default (unless
    explicitly marked as ``"mutable": False`` in
    ``_CONFIG_DEFINITIONS``). Once the feature is stabilized, the default
    will be changed to **immutable**.

    Note that updating a configuration only modifies the value in the
    ``LMCacheEngineConfig`` object. If a component has already read and
    cached the value elsewhere (e.g., stored in a local variable or
    another object during initialization), the change will **not** take
    effect for that component.

The request body should be a JSON object with config name-value pairs.
Type conversion is handled automatically (e.g., string ``"512"`` will
be converted to integer ``512`` based on the config definition).

The response contains an ``updated`` field with successfully applied
values, and an ``errors`` field if any keys failed:

.. code-block:: json

    {
      "updated": {"min_retrieve_tokens": 512, "save_decode_cache": true},
      "errors": {"unknown_key": "Unknown config"}
    }

`/meta` endpoint for metadata:

.. code-block:: bash

    curl http://localhost:7000/meta

`/threads` endpoint for threads:

.. code-block:: bash

    curl http://localhost:7000/threads

`/loglevel` endpoint for log level:

.. code-block:: bash

    # Get all loggers info
    curl http://localhost:7000/loglevel
    # Get specified logger level
    curl http://localhost:7000/loglevel?logger_name=lmcache.v1.cache_engine
    # Set specified logger level
    curl http://localhost:7000/loglevel?logger_name=lmcache.v1.cache_engine&level=DEBUG

`/run_script` endpoint for running script:

.. code-block:: bash

    curl -X POST http://localhost:7000/run_script \
      -F "script=@/Users/msy/scratch.py"

    {'is_first_rank': True, 'model_version': (27, 1, 64, 1, 576), 'LocalCPUBackend.use_hot': False}

`scratch.py`:

.. code-block:: python
    
    # Get cache_engine from app.state
    lmcache_engine = app.state.lmcache_adapter.lmcache_engine

    # Print the worker ID and model name
    print(f"Worker ID: {lmcache_engine.metadata.worker_id}")
    print(f"Model name: {lmcache_engine.metadata.model_name}")

    # Set LocalCPUBackend.use_hot to False or True
    lmcache_engine.storage_manager.storage_backends["LocalCPUBackend"].use_hot = False
    # return the output contents
    result = {
        "is_first_rank": lmcache_engine.metadata.is_first_rank(),
        "model_version": lmcache_engine.metadata.kv_shape,
        "LocalCPUBackend.use_hot": lmcache_engine.storage_manager.storage_backends["LocalCPUBackend"].use_hot
    }

How to extend the Internal API Server
=======================================

You can extend the ``internal_api_server`` by adding new endpoint files to the `lmcache/v1/internal_api_server/` directory.
Ensure your new file name ends with `_api.py`. Additionally, you need to define a `router = APIRouter()` in your file and add your endpoints to it.
