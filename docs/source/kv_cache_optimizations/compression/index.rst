Compression
===========

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the MP mode equivalent of this page, see :doc:`/mp/cachegen`.


KV cache compression can greatly reduces the size of the cache, which can be beneficial for both storage/memory usage and loading speed.
Currently, we support the following compression algorithms:

- :ref:`CacheGen <cachegen>`: `CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving <https://dl.acm.org/doi/10.1145/3651890.3672274>`_


.. toctree::
   :maxdepth: 1

   cachegen
