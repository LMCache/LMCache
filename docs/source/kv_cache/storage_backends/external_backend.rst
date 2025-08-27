External Storage Backends
=======================

LMCache supports integrating custom storage backends through dynamic loading. This allows extending cache storage capabilities without modifying core code.

Configuration
-------------
Add the following to your ``lmcache.yaml``:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    external_backends: "log_external_backend"
    extra_config:
      external_backend.log_external_backend.module_path: lmc_external_log_backend.lmc_external_log_backend
      external_backend.log_external_backend.class_name: ExternalLogBackend

Implementation Example
----------------------
A sample backend implementation (e.g., https://github.com/opendataio/lmc_external_log_backend/ ):

Key Requirements
----------------
1. Inherit from ``StorageBackendInterface``
2. Implement all abstract methods
3. Provide the constructor as this example
4. Package as installable Python module

Usage Notes
-----------
1. Install your backend package in LMCache environment
2. Add ``external_backends`` configuration and its related ``module_path`` and  ``class_name`` to ``extra_config`` section
3. Backends are initialized during LMCache startup
4. Use ``external_backends`` list to enable specific backends

.. note::
   Backends are loaded in order - earlier backends have higher priority during cache lookups.