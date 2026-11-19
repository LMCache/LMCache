MP Release Notes
================

Unreleased
----------

KV-cache STORE completeness
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LMCache-driven MP STORE and engine-driven multigroup STORE paths skip
an object if any block they would copy is absent. This prevents
incomplete recurrent-state or sliding-window data from being published under
valid token hashes and reused by a later prefix-cache hit.

When groups are merged with ``--no-separate-object-groups``, an absent STATE
slice also prevents storing otherwise complete PAGE data in that object. The
default ``--separate-object-groups`` keeps complete PAGE objects independent
for full chunks. A partial final chunk still requires all groups. See
:doc:`hybrid_models` for the tradeoff.

.. warning::

   Existing invalid entries are not detected, repaired, or invalidated by this
   fix. Affected deployments must restart their services with the fixed
   version and rebuild caches using fresh namespaces or storage locations on
   all persistent backends. Restarting alone does not remove old persistent
   objects. Follow the migration steps in :doc:`deployment` before reusing
   the cache service.
