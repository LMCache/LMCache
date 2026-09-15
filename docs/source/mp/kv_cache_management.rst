.. _mp_kv_cache_management:

KV Cache Management
===================

In multiprocess (MP) mode, cache management is a **fleet-level** operation
served by the coordinator's ``/cache`` group: warm prefetch, pin/unpin,
delete, and move, each addressed by token ids and dispatched to the named
MP server(s). See :ref:`Cache control <mp_coordinator_cache_control>` on the
:doc:`coordinator` page.

.. note::

   The in-process management commands (including ``compress``, which has no
   MP management endpoint) are documented in the
   :doc:`Legacy section <../legacy/index>`:
   :doc:`../kv_cache_management/index`.
