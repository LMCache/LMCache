.. WORDPRESS-SYNC metadata block. Sphinx ignores this comment.

   slug: glossary
   wp_post_type: page
   wp_status: draft
   canonical: https://lmcache.ai/glossary
   schema_type: DefinedTermSet
   seo_title: LLM Inference and KV Cache Glossary | LMCache
   meta_description: Plain-language definitions of the inference, caching, benchmarking, and deployment terms used across LMCache.
   last_reviewed: 2026-09-10

.. _glossary:

Glossary
========

.. meta::
   :description: Plain-language definitions of the inference, caching, and benchmarking terms used across LMCache.
   :keywords: KV cache glossary, LLM inference glossary, TTFT, prefill, decode

Definitions of the terms that come up when reading, writing, or benchmarking
anything about LMCache. Every entry stands alone: land on one cold, understand
it without reading anything above it, and follow the links outward from there.

For the infrastructure vocabulary that appears in deployment discussions --
pods, nodes, replicas, probes -- see :doc:`kubernetes`.

Inference primitives
--------------------

How a model turns text into text. The foundation everything else rests on.

:doc:`Inference <inference>`
   Inference is the process of running a trained model to produce an output — as opposed to
   training, which is the process of producing the model in the...

:doc:`Token <token>`
   A token is the unit of text a model actually operates on — a whole word, a piece of a
   word, punctuation, whitespace, or another recurring fragment,...

:doc:`Tokenization <tokenization>`
   Tokenization is the step that splits raw text into tokens the model recognizes.

:doc:`Embedding <embedding>`
   An embedding is the vector — a list of numbers — that represents a token inside the
   model.

:doc:`Model weights <model-weights>`
   Model weights are the numbers a model learned during training and keeps fixed during
   inference.

:doc:`Attention <attention>`
   Attention is the mechanism that lets each token in a sequence draw information from the
   other tokens around it.

:doc:`Query, Key, and Value (QKV) <query-key-value>`
   Query, Key, and Value are three vectors derived from each token's current
   representation, and together they work like a lookup.

:doc:`Attention head <attention-head>`
   An attention head is one independent copy of the attention computation, running on its
   own slice of each token's representation.

:doc:`Grouped-query attention (GQA) <grouped-query-attention>`
   Grouped-query attention is an architecture choice where several Query heads share a
   single Key/Value head, rather than each Query head having its own.

:doc:`Causal mask <causal-mask>`
   A causal mask blocks a token from attending to positions that come after it, enforcing
   that a model generating text left to right can use itself and...

:doc:`Autoregressive generation <autoregressive-generation>`
   Autoregressive generation is the loop a model runs to produce text: generate one token,
   append it to the sequence, then use the longer sequence to...

:doc:`Context length <context-length>`
   Context length is the maximum number of tokens a model can attend over in a single
   request — prompt plus generated output together.

Prefill, decode, and the KV cache
---------------------------------

The specific problem LMCache exists to solve.

:doc:`Prefill <prefill>`
   Prefill is the phase where the model processes the prompt, before it generates anything.

:doc:`Decode <decode>`
   Decode is the phase where the model generates output, one token at a time.

:doc:`KV cache <kv-cache>`
   A KV cache is the stored collection of Key and Value tensors a model has already
   computed for the tokens it has seen, kept so those tokens never have...

:doc:`Prefix caching <prefix-caching>`
   Prefix caching is reusing a KV cache across separate requests, not just within one
   generation.

:doc:`Cache hit / cache miss <cache-hit>`
   A cache hit means the key/value tensors for a stretch of tokens were already computed
   and can be reused; a cache miss means there is nothing to reuse...

:doc:`Chunk size <chunk-size>`
   Chunk size is the granularity at which cached KV data is stored and looked up — 256
   tokens by default in LMCache (chunk_size).

:doc:`Eviction policy <eviction-policy>`
   An eviction policy decides what to discard when a cache pool is full.

:doc:`Cache tiering and offloading <cache-tiering>`
   Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower
   storage, so it stays reusable instead of being thrown away.

:doc:`CacheBlend <cacheblend>`
   CacheBlend is an LMCache technique for reusing cached KV data across chunks that are
   *not* in the position they were cached in — most usefully in RAG...

:doc:`Prefill–decode (P/D) disaggregation <pd-disaggregation>`
   P/D disaggregation is running prefill and decode on separate GPU pools or machines
   instead of both phases on the same worker.

:doc:`Cache-aware routing <cache-aware-routing>`
   Cache-aware routing sends requests likely to share a prefix to the same replica, so the
   cached data they need is actually reachable.

:doc:`LMCache <lmcache>`
   LMCache is an open-source KV-cache layer for LLM serving.

Performance metrics
-------------------

The vocabulary of any benchmark claim.

:doc:`Time to first token (TTFT) <ttft>`
   TTFT is the delay between a request arriving and the model producing its first output
   token.

:doc:`Inter-token latency (ITL / TPOT) <inter-token-latency>`
   ITL is the delay between each subsequent output token once generation has started.

:doc:`Interactivity <interactivity>`
   Interactivity is tokens delivered per second per user — roughly the inverse of inter-
   token latency.

:doc:`Tail latency (p50 / p90 / p99) <tail-latency>`
   Tail latency is the latency experienced by the slowest requests, reported as percentiles
   rather than an average.

:doc:`Decode throughput <decode-throughput>`
   Decode throughput is the aggregate rate of output tokens per second across all
   concurrent requests.

:doc:`Prefill throughput <prefill-throughput>`
   Prefill throughput is the aggregate rate at which the system processes prompt tokens, in
   tokens per second.

:doc:`QPS (queries per second) <qps>`
   QPS is completed requests per second.

:doc:`Cache hit rate <cache-hit-rate>`
   Cache hit rate is the share of requested tokens that were served from cache rather than
   recomputed.

:doc:`Concurrency and batching <concurrency>`
   Concurrency is how many requests a server works on at once; batching is how it groups
   them so the GPU processes them together.

:doc:`Total cost of ownership (TCO) <total-cost-of-ownership>`
   TCO is the cost of running a deployment over its lifetime, not its sticker price.

Engine and hardware knobs
-------------------------

The settings that move the metrics, most of which are not LMCache's.

:doc:`Quantization and precision <quantization>`
   Quantization is storing and computing numbers at lower precision to save memory and
   time.

:doc:`Tensor parallelism (TP) <tensor-parallelism>`
   Tensor parallelism splits each layer's weight matrix across several GPUs, so each GPU
   computes part of one matrix multiplication and the GPUs then...

:doc:`Pipeline parallelism (PP) <pipeline-parallelism>`
   Pipeline parallelism assigns different whole layers to different GPUs, so a token's
   forward pass flows through them like an assembly line — GPU 0...

:doc:`Expert parallelism (EP) <expert-parallelism>`
   Expert parallelism places different experts of a mixture-of-experts model on different
   GPUs, routing each token to whichever GPU holds the expert its...

:doc:`Speculative decoding <speculative-decoding>`
   Speculative decoding uses a smaller "draft" model to propose several tokens ahead, which
   the main model then verifies in a single pass.

:doc:`Attention kernel <attention-kernel>`
   A kernel is a low-level GPU program implementing one specific computation as hardware-
   optimized code; the attention kernel is the code that actually...

:doc:`Interconnects (PCIe, NVLink, RDMA) <interconnects>`
   Interconnects are the links data travels over between GPUs, machines, and storage.

.. toctree::
   :hidden:
   :maxdepth: 1

   inference
   token
   tokenization
   embedding
   model-weights
   attention
   query-key-value
   attention-head
   grouped-query-attention
   causal-mask
   autoregressive-generation
   context-length
   prefill
   decode
   kv-cache
   prefix-caching
   cache-hit
   chunk-size
   eviction-policy
   cache-tiering
   cacheblend
   pd-disaggregation
   cache-aware-routing
   lmcache
   ttft
   inter-token-latency
   interactivity
   tail-latency
   decode-throughput
   prefill-throughput
   qps
   cache-hit-rate
   concurrency
   total-cost-of-ownership
   quantization
   tensor-parallelism
   pipeline-parallelism
   expert-parallelism
   speculative-decoding
   attention-kernel
   interconnects
   kubernetes
