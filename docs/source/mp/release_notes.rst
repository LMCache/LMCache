MP Release Notes
================

Unreleased
----------

KV-cache STORE completeness
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LMCache-driven MP STORE path now checks presence after window downsampling
and skips an object if any required retained block is absent. This prevents
incomplete recurrent-state or sliding-window data from being published under
valid token hashes and reused by a later prefix-cache hit.

In the default merged configuration, an absent STATE slice also prevents
storing otherwise complete PAGE data in that object. Use
``--separate-object-groups`` to retain complete PAGE objects independently;
see :doc:`hybrid_models` for the tradeoff.

.. warning::

   Existing invalid entries are not detected, repaired, or invalidated by this
   fix. Affected deployments must restart their services with the fixed
   version and rebuild caches using fresh namespaces or storage locations on
   all persistent backends. Restarting alone does not remove old persistent
   objects. Follow the migration steps in :doc:`deployment` before reusing
   the cache service.
