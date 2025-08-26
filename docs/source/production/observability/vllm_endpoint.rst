.. _observability_vllm_endpoint:

Metrics by vLLM API
==========================================

LMCache provides detailed metrics via a Prometheus endpoint, allowing for in-depth monitoring of cache performance and behavior.
This section outlines how to enable and configure observability from embedded vLLM ``/metrics`` API endpoint.


Steps
-----

1) On vLLM/LMCache side
^^^^^^^^^^^^^^^^^^^^^^^

In v1, vLLM and LMCache run in separate processes, so you have to use multi‑process Prometheus.

The ``PROMETHEUS_MULTIPROC_DIR`` environment variable must be the same in both processes, as a IPC directory.

.. code-block:: bash

   PROMETHEUS_MULTIPROC_DIR=/tmp/lmcache_prometheus \
   #.. other environment variables \
   vllm serve $MODEL -port 8000 ...

Once the HTTP server is running, you can access the LMCache metrics at the ``/metrics`` endpoint.

.. code-block:: bash

   curl http://$<vllm-worker-ip>:8000/metrics | grep lmcache

   # Replace $IP with the IP address of a vLLM worker


And you will also find some ``.db`` files in the ``$PROMETHEUS_MULTIPROC_DIR`` directory.


2) Prometheus Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To scrape the LMCache metrics with a Prometheus server, add the following job to your ``prometheus.yml`` configuration,
or equivalent configuration to scrape the metrics endpoint:

.. code-block:: yaml

   scrape_configs:
     - job_name: 'lmcache'
       static_configs:
         - targets: ['<vllm-worker-ip>:8000']
       scrape_interval: 15s

Available Metrics
-----------------

LMCache exposes a variety of metrics to monitor its performance, including:

- ``lmcache:num_retrieve_requests``: Total number of retrieve requests.
- ``lmcache:num_store_requests``: Total number of store requests.
- ``lmcache:num_lookup_requests``: Total number of lookup requests.
- ``lmcache:num_requested_tokens``: Total number of tokens requested for retrieval.
- ``lmcache:num_hit_tokens``: Total number of cache hit tokens from retrieval.
- ``lmcache:retrieve_hit_rate``: The hit rate for retrieve requests.
- ``lmcache:lookup_hit_rate``: The hit rate for lookup requests.
- ``lmcache:local_cache_usage``: Local cache usage in bytes.
- ``lmcache:remote_cache_usage``: Remote cache usage in bytes.
- ``lmcache:time_to_retrieve``: A histogram of time taken to retrieve from the cache (seconds).
- ``lmcache:time_to_store``: A histogram of time taken to store to the cache (seconds).
- ``lmcache:retrieve_speed``: A histogram of retrieval speed (tokens per second).
- ``lmcache:store_speed``: A histogram of storage speed (tokens per second).


