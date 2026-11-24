CacheBlend
==========

CacheBlend lets LMCache reuse the KV cache of **any** repeated text chunk --
not only a shared prefix -- by selectively recomputing a small fraction of
tokens at chunk boundaries. This cuts time-to-first-token for RAG and
multi-document workloads where the reusable context is not a clean prefix.

Enabling CacheBlend (MP mode)
-----------------------------

Start the LMCache server with the blend engine:

.. code-block:: bash

   lmcache server --l1-size-gb 20 --eviction-policy LRU --engine-type blend

The ``blend`` engine composes a ``BlendModule`` into the server and requires
``--supported-transfer-mode`` to be ``lmcache_driven`` (the default) or ``auto``. See
:doc:`/mp/configuration` for the related server flags.

.. note::

   The legacy in-process CacheBlend documentation is maintained in
   `LMCache_inprocess <https://github.com/LMCache/LMCache_inprocess>`_.
