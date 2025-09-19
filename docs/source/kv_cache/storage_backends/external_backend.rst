External Storage Backends
=========================

LMCache supports integrating custom storage backends through dynamic loading. This allows extending cache storage capabilities without modifying core code.

Backend Definition Requirements
-------------------------------
1. Inherit from ``StorageBackendInterface``
2. Add constructor with the same signature as ``StorageBackendInterface``
3. Implement all abstract methods
4. Package as an installable Python module

How to Integrate the Backend with LMCache
-----------------------------------------
1. Install your backend package in the LMCache environment
2. Add ``external_backends`` and its related ``module_path`` and ``class_name`` to ``extra_config`` section of LMCache configuration as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    external_backends: <backend_name>
    extra_config:
      external_backend.<backend_name>.module_path: <module_path>
      external_backend.<backend_name>.class_name: <class_name>

An example configuration for a logging backend is as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    external_backends: "log_external_backend"
    extra_config:
      external_backend.log_external_backend.module_path: lmc_external_log_backend.lmc_external_log_backend
      external_backend.log_external_backend.class_name: ExternalLogBackend

.. note::

   - Backends are initialized in order during LMCache startup - earlier backends have higher priority during cache lookups
   - ``external_backends.<backend_name>`` distinguishes the different dynamic loaded backends

Backend Implementation Example
------------------------------
A sample backend implementation can be viewed at https://github.com/opendataio/lmc_external_log_backend/

