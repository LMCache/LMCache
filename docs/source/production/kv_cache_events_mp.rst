KV Cache Events
===============

.. note::

   Multiprocess (MP) support for KV cache events is **planned (TODO)**. The
   current in-process implementation (vLLM ``--kv-events-config`` +
   ``enable_kv_events``) is preserved in the Legacy section:
   :doc:`/production/kv_cache_events`.

   In MP today, lifecycle and telemetry events are available through the
   :doc:`/mp/observability` EventBus instead.
