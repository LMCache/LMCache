Storage Extension Interfaces
============================

Storage APIs are separate for the two deployment modes:

- MP backends implement ``L2AdapterInterface`` under
  ``lmcache.v1.distributed.l2_adapters``. See
  :doc:`/developer_guide/extending_lmcache/storage_plugins` and
  :doc:`/developer_guide/extending_lmcache/native_connectors` for development,
  and :doc:`/mp/l2_storage/index` for deployment.
- Deprecated in-process backends implement ``StorageBackendInterface`` or
  ``StoragePluginInterface`` under ``lmcache.v1.storage_backend``. See
  :doc:`/legacy/developer_guide/extending_lmcache/storage_plugins` and
  :doc:`/legacy/developer_guide/extending_lmcache/remote_storage_plugins`.
