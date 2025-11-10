Configurable Storage Backends
=============================

LMCache supports integrating custom storage backends through dynamic loading or plug and play capability. This allows extending cache storage capabilities without modifying core code.

Backend Definition Requirements
-------------------------------
1. Inherit from ``ConfigurableStorageBackendInterface``
2. Implement all the abstract methods of the parent interface of ``ConfigurableStorageBackendInterface``- ``StorageBackendInterface``
3. Package as an installable Python module

.. note::

  The interface constructor is the instantiation contract that the LMCache loading system will use when loading configurable storage backends.
  If you wish to implement a constructor, it should have the same parameter signature and call the interface constructor.

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

