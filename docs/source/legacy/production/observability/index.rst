.. _observability:

Observability
=============

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the related MP documentation and its separate configuration, see :doc:`/mp/observability/index`.


.. toctree::
   :maxdepth: 2

   vllm_endpoint
   internal_api_server
   metrics
   chunk_statistics
   health_monitor
   frontend

Periodic-thread diagnostics are shared across modes; see
:doc:`/production/observability/periodic_thread_api`.
