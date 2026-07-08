.. _usage-stats-collection:

Usage Stats Collection
======================

LMCache collects anonymous usage data by default to help the engineering team understand real-world workloads, prioritize optimizations, and improve reliability. All collected data is aggregated and contains no sensitive user information.

A sanitized subset of the aggregated data may be publicly released for the community’s benefit (for example, see a daily usage report `here <https://github.com/Hanchenli/OSS_Growth_Toolkit/tree/main/usage_tracker/report>`_).

What data is collected?
-----------------------

Usage stats are emitted as three message types, implemented in the ``lmcache/usage_telemetry/`` package:

- **EnvMessage**  
  Captures environment details such as cloud provider, CPU info, total memory, architecture, GPU count/type, and execution source.  

- **EngineMessage**  
  Records engine configuration and metadata, including cache settings (chunk size, local device, cache limits), remote backend parameters, blending settings, model name, world size, and KV-cache dtype/shape.  

- **MetadataMessage**  
  Reports execution metadata: the timestamp when the run started and total duration in seconds.

In addition to the one-shot messages above, a continuous reporter periodically
sends interval counters (**ContinuousContextMessage**: tokens stored/hit and
stored KV bytes in the interval) and a cache-lifespan histogram
(**CacheLifespanMessage**). The flush interval is controlled by
``LMCACHE_USAGE_TRACK_INTERVAL`` (seconds, default 600).

Every payload carries three correlation fields:

- ``session_id`` -- a random UUID minted once per process, joining the
  one-shot context with the continuous messages of the same run.
- ``machine_id`` -- a random UUID persisted at
  ``~/.config/lmcache/machine_id``, grouping sessions from the same machine.
  It is never derived from hardware identifiers (MAC address, hostname).
- ``schema_version`` -- the version of the message schema.

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

By default, usage tracking is **enabled**. Any one of the following opt-outs
disables all usage stats collection:

.. code-block:: bash

   # LMCache-specific opt-out
   export LMCACHE_TRACK_USAGE=false

   # The cross-tool "do not track" convention (1/true/yes)
   export DO_NOT_TRACK=1

   # Or a persistent marker file, no environment needed
   mkdir -p ~/.config/lmcache && touch ~/.config/lmcache/do_not_track

When tracking is disabled, ``InitializeUsageContext`` will return ``None`` and
no data will be sent or logged, and no state files (such as ``machine_id``)
will be created.

Local logging
~~~~~~~~~~~~~

If you would like to log to a file in addition to (or instead of) sending data to the server, pass a local-log path when initializing:

.. code-block:: python

   from lmcache.usage_telemetry import InitializeUsageContext

   usage_ctx = InitializeUsageContext(
       config=engine_config,
       metadata=engine_metadata,
       local_log="~/.config/lmcache/usage.log"
   )

Omitting the ``local_log`` argument (or passing ``None``) disables local file logging.
