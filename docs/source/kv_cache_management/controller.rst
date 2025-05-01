LMCache Controller
==================

LMCache Controller exposes a set of APIs for users and orchestrators to manage the KV cache.

Currently, the controller provides the following APIs:

- :ref:`Lookup <lookup>`: Lookup the KV cache for a given list of tokens.
- :ref:`Clear <clear>`: Clear the KV caches.
- :ref:`Pin <pin>`: Persist or set the TTL of a KV cache.
- :ref:`Move <move>`: Move the KV cache to a different location.
- :ref:`Compress <compress>`: Compress the KV cache.
- :ref:`CheckFinish <check_finish>`: Check whether a (non-blocking) control event has finished or not.
