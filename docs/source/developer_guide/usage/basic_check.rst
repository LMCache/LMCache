Basic Check Tool
================

``python -m lmcache.v1.basic_check`` has separate checks for MP L2 adapters
and deprecated in-process storage backends. Run in an installed LMCache
Python environment with the dependencies of the selected backend.

MP L2 adapter check
-------------------

The ``test_l2_adapter`` mode constructs an ``L2AdapterInterface`` directly
and validates store, lookup, load, and data integrity with ``ObjectKey`` keys.
It does not require a running MP server or an inference engine.

.. code-block:: bash

   python -m lmcache.v1.basic_check --mode test_l2_adapter \
     --l2-adapter '{"type":"mock","max_size_gb":1}' \
     --num-keys 5 --obj-size 1024 --kv-dtype float32

Repeat ``--l2-adapter`` to check multiple adapters. Use ``--settle-time``
for backends requiring a delay between store and lookup. See
:doc:`/mp/l2_storage/supported_storages` for adapter configuration.

Other modes
-----------

.. code-block:: bash

   python -m lmcache.v1.basic_check --mode list

The ``test_remote``, ``test_storage_manager``, and ``gen`` modes use the
old storage backend interfaces. Their YAML configuration and examples are
kept in :doc:`/legacy/developer_guide/usage/basic_check`.
