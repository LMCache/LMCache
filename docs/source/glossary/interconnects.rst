.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment; the
   glossary sync tool parses it. Keep the keys stable -- they are the
   contract between this file and the published page.

   slug: interconnects
   question: What are PCIe, NVLink, and RDMA?
   aliases: PCIe, NVLink, RDMA, interconnect, fabric
   related: cache-tiering, tensor-parallelism, pd-disaggregation, ttft
   tier: 4
   publish: yes
   wp_post_type: glossary
   wp_status: draft
   canonical: https://lmcache.ai/glossary/interconnects
   schema_type: DefinedTerm
   seo_title: What are PCIe, NVLink, and RDMA? | LMCache
   meta_description: Interconnects are the links data travels over between GPUs, machines, and storage.
   last_reviewed: 2026-09-10
   review_status: needs-technical-review

.. _glossary-interconnects:

Interconnects (PCIe, NVLink, RDMA)
==================================

.. meta::
   :description: Interconnects are the links data travels over between GPUs, machines, and storage.
   :keywords: Interconnects (PCIe, NVLink, RDMA), PCIe, NVLink, RDMA, interconnect, fabric

**Interconnects are the links data travels over between GPUs, machines, and storage.** The interconnect limits how fast cached KV data can move, so a benchmark result depends on which one the data crossed.

- PCIe: the standard bus connecting a GPU to the rest of its machine (CPU, RAM). Speed depends on generation and lane count.
- NVLink: a much faster GPU-to-GPU interconnect than PCIe, available only on hardware that supports it. On some systems, such as NVIDIA's Grace-based superchips, a related link (NVLink-C2C) also connects the GPU to the CPU.
- RDMA: lets one machine read and write another machine's memory directly over the network, bypassing the remote CPU. Much faster and lower-latency than a conventional TCP/HTTP path. LMCache's peer-to-peer (P2P) KV cache sharing uses one-sided RDMA reads to pull cached data from another node's memory.

Any claim that cache retrieval is faster than recomputation is, underneath, a claim about which of these links the data crossed.

---

.. seealso::

   :doc:`cache-tiering` | :doc:`tensor-parallelism` | :doc:`pd-disaggregation` | :doc:`ttft`
