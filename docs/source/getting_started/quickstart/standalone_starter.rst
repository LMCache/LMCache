.. _standalone_starter:

Standalone Starter
==================

The LMCache Standalone Starter allows you to run LMCacheEngine as a standalone service without vLLM or GPU dependencies. This is particularly useful for:

- Testing and development environments
- CPU-only or P2P backend deployments

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: bash

   # Start with default configuration
   python -m lmcache.v1.standalone

   # Start with custom configuration file
   python -m lmcache.v1.standalone --config examples/cache_with_configs/example.yaml

   # Start with environment variables
   export LMCACHE_CONFIG_FILE=examples/cache_with_configs/example.yaml
   python -m lmcache.v1.standalone

CPU-Only Mode
~~~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --model_name my_model \
       --worker_id 0 \
       --world_size 1

Remote P2P Mode
~~~~~~~~~~~~~
TO be added

Configuration Section
---------------------

The standalone starter supports multiple configuration sources with the following priority order:

1. **Command-line arguments** (highest priority)
2. **Configuration file** (specified by ``--config`` or ``LMCACHE_CONFIG_FILE``)
3. **Environment variables** (e.g., ``LMCACHE_CHUNK_SIZE=512``)
4. **Default values** (lowest priority)

Command-Line Parameters
-----------------------

Basic Parameters
~~~~~~~~~~~~~~~~

.. code-block:: bash

   --config CONFIG_FILE             # Path to configuration file
   --model_name MODEL_NAME          # Model name for cache identification
   --worker_id WORKER_ID            # Worker ID (default: 0)
   --world_size WORLD_SIZE          # Total workers (default: 1)
   --kv_dtype {float16,float32,bfloat16}  # KV cache data type


Usage Examples
--------------

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --chunk_size=512 \
       --max_local_cpu_size=4.0 \
       --model_name=custom_model

Internal API Server
-------------------

The standalone starter includes an internal API server for monitoring and management:

.. code-block:: bash

   python -m lmcache.v1.standalone \
       --config examples/cache_with_configs/example.yaml \
       --chunk_size=512 \
       --max_local_cpu_size=4.0 \
       --model_name=custom_model \
       --internal_api_server_enabled=True


Troubleshooting
----------------

Common Issues
~~~~~~~~~~~~~

**Issue**: "No config file specified"
**Solution**: Set ``LMCACHE_CONFIG_FILE`` or use ``--config`` parameter

**Issue**: "Failed to connect to controller"
**Solution**: Start controller first: ``python -m lmcache.v1.api_server``

Debug Mode
~~~~~~~~~~

Enable debug logging for troubleshooting:

.. code-block:: bash

   export LMCACHE_LOG_LEVEL=DEBUG
   python -m lmcache.v1.standalone

Performance Tuning
------------------

Memory Configuration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # For systems with large memory
   --max_local_cpu_size=8.0

   # For memory-constrained systems
   --max_local_cpu_size=1.0


Best Practices
--------------

1. **Use configuration files** for production deployments
2. **Set appropriate cache sizes** based on available memory
3. **Enable internal API** for monitoring and management
4. **Monitor logs** for performance and error tracking

Related Documentation
---------------------

- :doc:`../quickstart`
- :doc:`../../api_reference/configurations`
- :doc:`../../kv_cache/storage_backends/index`
- :doc:`../../kv_cache_management/index`