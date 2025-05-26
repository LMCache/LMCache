.. _usage-stats-collection:

Usage Stats Collection
======================

LMCache collects anonymous usage data by default to help the engineering team understand real-world workloads, prioritize optimizations, and improve reliability. All collected data is aggregated and contains no sensitive user information.

A sanitized subset of the aggregated data may be publicly released for the community’s benefit (for example, see a daily usage report `here <https://github.com/Hanchenli/OSS_Growth_Toolkit/tree/main/usage_tracker/report>`_).

What data is collected?
-----------------------

Usage stats are emitted as three message types, implemented in ``usage_context.py``:

- **EnvMessage**  
  Captures environment details such as cloud provider, CPU info, total memory, architecture, GPU count/type, and execution source.  

- **EngineMessage**  
  Records engine configuration and metadata, including cache settings (chunk size, local device, cache limits), remote backend parameters, blending settings, model name, world size, and KV-cache dtype/shape.  

- **MetadataMessage**  
  Reports execution metadata: the timestamp when the run started and total duration in seconds.

These messages are serialized to JSON and POSTed to the LMCache usage server.

Example JSON payload
~~~~~~~~~~~~~~~~~~~~

.. code-block:: json

   {
     "message_type": "EnvMessage",
     "provider": "GCP",
     "num_cpu": 24,
     "cpu_type": "Intel(R) Xeon(R) CPU @ 2.20GHz",
     "cpu_family_model_stepping": "6,85,7",
     "total_memory": 101261135872,
     "architecture": ["64bit", "ELF"],
     "platforms": "Linux-5.10.0-28-cloud-amd64-x86_64-with-glibc2.31",
     "gpu_count": 2,
     "gpu_type": "NVIDIA L4",
     "gpu_memory_per_device": 23580639232,
     "source": "DOCKER"
   }

Previewing collected data
-------------------------

If you enable **local logging**, usage messages are appended to your specified log file. To inspect the most recent entries:

.. code-block:: bash

   tail ~/.config/lmcache/usage.log

Configuration & Opt-out
-----------------------

By default, usage tracking is **enabled**. To disable all usage stats collection, set the environment variable:

.. code-block:: bash

   export LMCACHE_TRACK_USAGE=false

When ``LMCACHE_TRACK_USAGE`` is set to ``false``, ``InitializeUsageContext`` will return ``None`` and no data will be sent or logged.

Local logging
~~~~~~~~~~~~~

If you would like to log to a file in addition to (or instead of) sending data to the server, pass a local-log path when initializing:

.. code-block:: python

   from lmcache.usage_context import InitializeUsageContext

   usage_ctx = InitializeUsageContext(
       config=engine_config,
       metadata=engine_metadata,
       local_log="~/.config/lmcache/usage.log"
   )

Omitting the ``local_log`` argument (or passing ``None``) disables local file logging.
