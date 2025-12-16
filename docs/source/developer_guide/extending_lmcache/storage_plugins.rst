Storage Plugins
===============

LMCache supports out of the box storage backends like Mooncake, S3 and NIXL.
The LMCache storage plugin system provides the ability to add custom storage backends through dynamic loading or plug and play capability. In other words, extending cache storage capabilities without modifying core code.

Backend Definition Requirements
-------------------------------
1. Inherit from ``StoragePluginInterface``
2. Implement all the abstract methods of the parent interface of ``StoragePluginInterface``- ``StorageBackendInterface``
3. Package as an installable Python module

.. note::

  The interface constructor is the instantiation contract that the LMCache loading system will use when loading custom storage backends.
  If you wish to implement a constructor, it should have the same parameter signature and call the interface constructor.

How to Integrate the Backend with LMCache
-----------------------------------------
1. Install your backend package in the LMCache environment
2. Add ``storage_plugins`` and its related ``module_path`` and ``class_name`` to ``extra_config`` section of LMCache configuration as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    storage_plugins: <backend_name>
    extra_config:
      storage_plugin.<backend_name>.module_path: <module_path>
      storage_plugin.<backend_name>.class_name: <class_name>

An example configuration for a logging backend is as follows:

.. code-block:: yaml

    chunk_size: 64
    local_cpu: False
    max_local_cpu_size: 5
    storage_plugins: "log_backend"
    extra_config:
      storage_plugin.log_backend.module_path: lmc_external_log_backend.lmc_external_log_backend
      storage_plugin.log_backend.class_name: ExternalLogBackend

.. note::

   - Storage backends are initialized in order during LMCache startup - earlier backends have higher priority during cache lookups
   - ``storage_plugin.<backend_name>`` distinguishes the different dynamic loaded backends

Backend Implementation Example
------------------------------
A sample custom backend implementation can be viewed at https://github.com/opendataio/lmc_external_log_backend/

