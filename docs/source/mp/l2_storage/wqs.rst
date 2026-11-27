WQS
===

An L2 adapter backed by `WQS <https://www.expontech.com/en/pages/product-display/wqs>`_ --
a high-performance distributed KV cache storage system from ExponTech.
WQS combines RDMA networking, SPDK-managed NVMe storage, and
hugepage shared-memory segments to store KV cache chunks at high throughput
and low latency, and allows a cluster of vLLM/LMCache instances to share a
single L2 cache pool.

.. note::

   The ``wqs_store`` adapter is developed and maintained by ExponTech and
   ships in their LMCache builds; it is not part of the open-source LMCache
   source tree.

For installation instructions, LMCache configuration guides, and
benchmark results, see the product page:

- https://www.expontech.com/en/pages/product-display/wqs
