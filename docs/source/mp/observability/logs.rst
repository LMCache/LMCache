Logging
=======

Logging subscribers emit debug-level messages for store, retrieve, lookup,
L1, and StorageManager events via Python's standard ``logging`` module.

When OpenTelemetry is installed, ``init_logger`` automatically attaches an
OTel ``LoggingHandler`` so that log records are forwarded to any configured
OTel ``LoggerProvider``. The handler respects the ``LMCACHE_LOG_LEVEL``
environment variable.

.. code-block:: bash

    LMCACHE_LOG_LEVEL=DEBUG lmcache server ...

Key log messages:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Level
     - Message
   * - INFO
     - ``Stored N tokens in X seconds``
   * - INFO
     - ``Retrieved N tokens in X seconds``
   * - INFO
     - ``Prefetch request completed (L1+L2): N/M prefix hits``
   * - DEBUG
     - ``MP store start: session=... device=...``
   * - DEBUG
     - ``MP retrieve end: session=... retrieved_count=...``

