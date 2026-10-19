Using Different Storage Backends
================================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/l2_storage/index`.


LMCache supports various storage backends to offload and share KV cache data.

Supported Backends
-------------------------

.. toctree::
   :maxdepth: 1

   dax
   fs
   hfbucket
   resp
   s3
   
   
  
