.. _internal_api_server:

Internal API Server
===================

The ``internal_api_server`` provides HTTP APIs for managing and inspecting
the LMCache engine at runtime. APIs are organized into three categories:

- **Common APIs** — Available across all components (scheduler, worker, controller).
- **vLLM / Inference APIs** — Specific to vLLM inference workers.
- **Controller APIs** — Specific to the LMCache Controller.

.. toctree::
   :maxdepth: 2

   common_apis
   vllm_apis
   controller_apis


Configuration
-------------

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


Port Assignment
^^^^^^^^^^^^^^^

The port for each component is computed as:

.. code-block:: text

    actual_port = internal_api_server_port_start + port_offset

Where ``port_offset`` is:

- ``0`` for the Scheduler
- ``1 + worker_id`` for Workers (e.g., Worker 0 → offset 1, Worker 1 → offset 2)


API Category & Route Discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The server uses ``APIRegistry`` to automatically discover and register
API endpoint modules. Any file named ``*_api.py`` under
``lmcache/v1/internal_api_server/{common,vllm,controller}/`` that
exports a ``router = APIRouter()`` will be automatically included.


Extending the Server
^^^^^^^^^^^^^^^^^^^^^

To add a new API endpoint:

1. Create a new file in the appropriate category directory
   (``common/``, ``vllm/``, or ``controller/``).
2. Name the file with ``_api.py`` suffix (e.g., ``my_feature_api.py``).
3. Define ``router = APIRouter()`` and add your endpoints.

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

.. _bypass_mode:

Bypass Mode
-----------

Bypass mode allows dynamically skipping specific storage backends at runtime.
Bypassed backends are excluded from ``contains``/``put``/``get`` operations.
This is useful for fault injection testing, isolating a problematic backend,
or debugging without restarting the engine.

``GET /bypass/list`` — List Bypassed Backends:

.. code-block:: bash

    curl http://localhost:7000/bypass/list

Example response:

.. code-block:: json

    {
      "status": "success",
      "bypassed_backends": ["RemoteBackend"],
      "all_backends": ["LocalCPUBackend", "RemoteBackend"]
    }

``PUT /bypass/add`` — Add a Backend to Bypass List:

.. code-block:: bash

    curl -X PUT "http://localhost:7000/bypass/add?backend_name=RemoteBackend"

Example response:

.. code-block:: json

    {
      "status": "success",
      "backend_name": "RemoteBackend",
      "bypassed": true,
      "was_already_bypassed": false,
      "bypassed_backends": ["RemoteBackend"]
    }

``PUT /bypass/remove`` — Remove a Backend from Bypass List:

.. code-block:: bash

    curl -X PUT "http://localhost:7000/bypass/remove?backend_name=RemoteBackend"

Example response:

.. code-block:: json

    {
      "status": "success",
      "backend_name": "RemoteBackend",
      "bypassed": false,
      "was_bypassed": true,
      "bypassed_backends": []
    }

Error response (unknown backend, HTTP 400):

.. code-block:: json

    {
      "error": "Unknown backend",
      "message": "Backend 'FooBackend' not found. Available: ['LocalCPUBackend', 'RemoteBackend']"
    }


How to extend the Internal API Server
=======================================

You can extend the ``internal_api_server`` by adding new endpoint files to the `lmcache/v1/internal_api_server/` directory.
Ensure your new file name ends with `_api.py`. Additionally, you need to define a `router = APIRouter()` in your file and add your endpoints to it.
>>>>>>> 8ee797df (Add bundle of bypass backend internal api apis)
=======
The endpoint will be automatically discovered and registered on the
next server startup.
=======
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

.. _bypass_mode:

Bypass Mode
-----------

Bypass mode allows dynamically skipping specific storage backends at runtime.
Bypassed backends are excluded from ``contains``/``put``/``get`` operations.
This is useful for fault injection testing, isolating a problematic backend,
or debugging without restarting the engine.

``GET /bypass/list`` — List Bypassed Backends:

.. code-block:: bash

    curl http://localhost:7000/bypass/list

Example response:

.. code-block:: json

    {
      "status": "success",
      "bypassed_backends": ["RemoteBackend"],
      "all_backends": ["LocalCPUBackend", "RemoteBackend"]
    }

``PUT /bypass/add`` — Add a Backend to Bypass List:

.. code-block:: bash

    curl -X PUT "http://localhost:7000/bypass/add?backend_name=RemoteBackend"

Example response:

.. code-block:: json

    {
      "status": "success",
      "backend_name": "RemoteBackend",
      "bypassed": true,
      "was_already_bypassed": false,
      "bypassed_backends": ["RemoteBackend"]
    }

``PUT /bypass/remove`` — Remove a Backend from Bypass List:

.. code-block:: bash

    curl -X PUT "http://localhost:7000/bypass/remove?backend_name=RemoteBackend"

Example response:

.. code-block:: json

    {
      "status": "success",
      "backend_name": "RemoteBackend",
      "bypassed": false,
      "was_bypassed": true,
      "bypassed_backends": []
    }

Error response (unknown backend, HTTP 400):

.. code-block:: json

    {
      "error": "Unknown backend",
      "message": "Backend 'FooBackend' not found. Available: ['LocalCPUBackend', 'RemoteBackend']"
    }


How to extend the Internal API Server
=======================================

You can extend the ``internal_api_server`` by adding new endpoint files to the `lmcache/v1/internal_api_server/` directory.
Ensure your new file name ends with `_api.py`. Additionally, you need to define a `router = APIRouter()` in your file and add your endpoints to it.
>>>>>>> 8ee797df (Add bundle of bypass backend internal api apis)
