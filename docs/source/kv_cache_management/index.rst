LMCache Controller
==================

LMCache Controller exposes a set of APIs for users and orchestrators to manage the KV cache.

Currently, the controller provides the following APIs:

- :ref:`Clear <clear>`: Clear the KV caches.
- :ref:`Compress <compress>`: Compress the KV cache.
- :ref:`Health <health>`: Check the health status of cache workers.
- :ref:`Lookup <lookup>`: Lookup the KV cache for a given list of tokens.
- :ref:`Move <move>`: Move the KV cache to a different location.
- :ref:`Pin <pin>`: Persist the KV cache to prevent it from being evicted.
- :ref:`CheckFinish <check_finish>`: Check whether a (non-blocking) control event has finished or not.

.. toctree::
   :maxdepth: 1
   :hidden:

   clear
   compress
   health
   lookup
   move
   pin
   check_finish
