Real-World Adoption
===================

Since its release, LMCache has rapidly grown into a key component of the LLM serving stack, attracting contributions and adoption from industry leaders.

Industry Adoption
-----------------

vLLM Production Stack
~~~~~~~~~~~~~~~~~~~~~

The official vLLM deployment tooling (for Kubernetes) includes LMCache out-of-the-box as the caching layer. This means that enterprise users deploying vLLM at scale have LMCache working behind the scenes to accelerate inference. The integration was motivated by substantial performance gains observed – for instance, internal benchmarks on ShareGPT conversation traces showed dramatically lower latency when LMCache was enabled, thanks to cross-user cache reuse.

llm-d by Red Hat/IBM
~~~~~~~~~~~~~~~~~~~~

LMCache is a core component of llm-d, a distributed inference project led by Red Hat (with IBM collaboration). Announced at Red Hat Summit 2025, llm-d uses LMCache for intelligent cache routing across clusters of vLLM servers. In this context, LMCache helps orchestrate KV sharing in a multi-LLM setup, ensuring that repeated content anywhere in the cluster can be served from a cache rather than recomputed. Red Hat's engineers have not only adopted LMCache but also joined as contributors to enhance its enterprise readiness.

NVIDIA Dynamo
~~~~~~~~~~~~~

NVIDIA's open-source Dynamo inference platform (built on vLLM) integrated LMCache as its KV cache solution in September 2025. By default, Dynamo lacked persistent caching (KV lived only in GPU memory per session); with LMCache, Dynamo can offload KV to external memory/storage and reuse it across queries and sessions. This was a milestone because it brought a battle-tested caching layer to a production-scale system used by many developers. The LMCache team worked closely with NVIDIA on this integration, also introducing the NiXL backend to fully leverage high-speed NVLink and RDMA in multi-GPU Dynamo deployments.

KServe (KFServing)
~~~~~~~~~~~~~~~~~~

LMCache is supported in KServe, the model serving platform for Kubernetes. This means organizations using KServe for LLMs can plug in LMCache to get caching benefits seamlessly in their inference clusters.

Community & Ecosystem
---------------------

The project has a vibrant community (5k+ stars on GitHub, as of Aug 2025) and is backed by research from UIUC and UChicago (the CacheGen/CacheBlend papers). Companies like IBM, Red Hat, Nvidia, AWS, and others have shown interest or contributed. For instance, IBM has used LMCache in internal stacks, and AWS Marketplace even lists LMCache as an accelerative layer for certain AI deployments. This broad interest underlines that KV caching is becoming a standard practice in LLM inference, with LMCache leading the charge.