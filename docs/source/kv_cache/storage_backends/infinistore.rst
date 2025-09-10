InfiniStore
===========

Coming soon...


.. _infinistore-overview:

Overview
--------

`InfiniStore <https://github.com/bytedance/InfiniStore>`_ is an open-source high-performance KV store and one of the remote KV storage options LMCache supports.

Infinistore supports RDMA and NVLink. LMCache's infinistore connector only uses RDMA transport.

InfiniStore Explanation:
------------------------

There are two major scenarios how InfiniStore supports:

Prefill-Decoding disaggregation clusters: in such mode inference workloads are separated into two node pools: prefill nodes and decoding nodes. InfiniStore enables KV cache transfer among these two types of nodes, and also KV cache reuse.
Non-disaggregated clusters: in such mode prefill and decoding workloads are mixed on every node. Infinistore serves as an extra large KV cache pool in addition to GPU cache and local CPU cache, and also enables cross-node KV cache reuse.

.. image:: ../../assets/InfiniStore-usage.png
    :alt: InfiniStore Usage Diagram


.. _infinistore-prerequisites:

Minimum Viable Example:
------------------------

To use InfiniStore as a remote RDMA-based backend for LMCache, you should have:

- Two bare metal machines on the same rack or data center network. Each machine must have a Mellanox RDMA-capable NIC, e.g., mlx5_0.

This minimal viable example will use OCI BM.GPU4.8 for LMCache + vLLM and BM.HPC2.36 for an InfiniStore backend.

Step 1: Create the InfiniStore server

Set up networking on the






