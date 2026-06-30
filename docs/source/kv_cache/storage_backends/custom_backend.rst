Custom Storage Backends
=======================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


LMCache supports integrating custom storage backends through dynamic loading or plug and play capability. This allows extending cache storage capabilities without modifying core code.

See :doc:`Storage Plugins <../../developer_guide/extending_lmcache/storage_plugins>` for more details.
