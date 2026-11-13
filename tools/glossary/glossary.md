---
title: LMCache Glossary
description: Plain-language definitions of the inference, caching, benchmarking, and deployment terms that come up around LMCache.
status: draft
version: 0.1
last_reviewed: 2026-09-10
sources:
  - handoff/Intro to KV Cache and LMCache -- DRAFT.pdf
  - handoff/benchmarks/metrics-and-factors.md
  - handoff/benchmarks/systems-vocabulary.md
maintainers:
  - TBD
---

# LMCache Glossary

Definitions of the terms that come up when reading, writing, or benchmarking anything about LMCache. Written for someone who is competent but new to this corner of the stack — an engineer arriving from a different specialty, a technical writer, an analyst, a journalist.

Every entry is written to stand alone. You should be able to land on one term cold, understand it without reading anything above it, and follow the `related` links outward if you want more.

## How this file is structured

Each term is a level-3 heading with a machine-readable metadata block above it:

```
<!--
slug: kv-cache
question: What is a KV cache?
aliases: [KV caching, key-value cache]
related: [prefill, decode, attention]
tier: 2
publish: yes
-->
```

- **`slug`** — the URL segment. One page per term: `lmcache.ai/glossary/<slug>`.
- **`question`** — the question-shaped form, for the page `<h1>`, the `<title>`, and FAQ schema.
- **`aliases`** — alternate spellings and expansions. Use for search, redirects, and auto-linking first mentions in blog posts.
- **`related`** — slugs of 3–6 sibling terms. These become the `relatedConcept` links; they are what turns a pile of definitions into a traversable graph.
- **`tier`** — which group the term belongs to (see below).
- **`publish`** — `yes` for terms that should get their own public page; `support` for terms that exist so other documents don't have to re-explain them inline, and that are better served as short entries on the index than as standalone pages.

The tiers:

1. **Inference primitives** — how a model turns text into text. The foundation everything else rests on.
2. **Prefill, decode, and the KV cache** — the specific problem LMCache exists to solve.
3. **Performance metrics** — the vocabulary of any benchmark claim.
4. **Engine and hardware knobs** — the settings that move the metrics, most of which are not LMCache's.
5. **Deployment and systems** — infrastructure terms that appear in deployment discussions.

A note carried over from the source material, worth keeping visible: LMCache's own configuration is never the only primary driver of any benchmark metric. Tiers 3, 4, and 5 exist largely so that claim can be understood.

---

## Tier 1: Inference primitives

<!--
slug: inference
question: What is inference?
aliases: [model inference, LLM inference]
related: [token, prefill, decode, model-weights]
tier: 1
publish: yes
-->
### Inference

**Inference is the process of running a trained model to produce an output** — as opposed to training, which is the process of producing the model in the first place.

The interface makes it look like words flow in and words flow out. Underneath, the hardware is moving and transforming large arrays of numbers: your prompt becomes numbers, those numbers pass through many layers of matrix multiplication, and the result is a set of probabilities for which token should come next. Those arrays occupy memory, usually GPU memory, and the intermediate results produced along the way have to live somewhere until the next operation needs them.

Once a model is deployed, the cost of running it in production is inference cost, because every request runs the model again. Much of that cost comes from memory and data movement rather than raw arithmetic: while the model generates output one token at a time, its speed is usually limited by how fast the GPU can read data from memory, not by how fast it can calculate.

<!--
slug: token
question: What is a token?
aliases: [tokens]
related: [tokenization, embedding, context-length, ttft]
tier: 1
publish: yes
-->
### Token

**A token is the unit of text a model actually operates on** — a whole word, a piece of a word, punctuation, whitespace, or another recurring fragment, drawn from the model's fixed vocabulary.

In English text, a token averages about four characters, or roughly three-quarters of a word; the ratio differs between tokenizers and between languages. Common words are usually a single token; rarer ones get split into pieces. Each token has an integer ID, which the model uses to look up its starting representation.

Tokens are the unit that nearly every other number in this glossary is denominated in: context length is measured in tokens, throughput is measured in tokens per second, KV cache size is measured in bytes per token, and API pricing is usually per million tokens.

<!--
slug: tokenization
question: What is tokenization?
aliases: [tokenizer, tokenizing]
related: [token, embedding, inference]
tier: 1
publish: yes
-->
### Tokenization

**Tokenization is the step that splits raw text into tokens the model recognizes.**

It happens before any model computation. The tokenizer is specific to the model, so the same sentence can produce a different number of tokens under different tokenizers. For that reason, token counts from one vendor's model do not carry over exactly to another vendor's model.

Tokenization can also complicate cost and context estimates: unusual formatting, some non-English text, and code or identifiers may use more tokens per character than ordinary English prose.

<!--
slug: embedding
question: What is an embedding?
aliases: [embeddings, embedding vector, vector]
related: [token, attention, tokenization]
tier: 1
publish: yes
-->
### Embedding

**An embedding is the vector — a list of numbers — that represents a token inside the model.** A model cannot multiply words, so every token is first converted into a fixed-length list of learned values it can operate on.

The mapping is built so that related tokens land near each other in that space. "Cat" and "dog" end up close together; "cat" and "car" do not, despite looking similar as text.

The initial embedding is only a starting point. As the vector passes through the model's layers, its representation shifts based on surrounding context. For example, the token "bank" contributes differently in "river bank" than in "bank account." The mechanism that lets one token take on information from others is [attention](/glossary/attention).

<!--
slug: model-weights
question: What are model weights?
aliases: [weights, parameters, model parameters]
related: [inference, quantization, attention, tensor-parallelism]
tier: 1
publish: yes
-->
### Model weights

**Model weights are the numbers a model learned during training and keeps fixed during inference.** They are the model itself; for example, an 8-billion-parameter model has 8 billion of them.

The important property for anything cache-related: weights do not grow with the prompt. A layer's projection matrices are the same size whether you send it four tokens or forty thousand. Everything that *does* grow with the prompt — the activations, the keys and values — is request-dependent data, and that is the part that has to be stored, moved, or thrown away.

This split between fixed weights and request-dependent tensors defines the KV-caching problem, because a cache only has to manage the request-dependent side.

<!--
slug: attention
question: What is attention in a transformer?
aliases: [attention mechanism, self-attention]
related: [query-key-value, attention-head, causal-mask, kv-cache]
tier: 1
publish: yes
-->
### Attention

**Attention is the mechanism that lets each token in a sequence draw information from the other tokens around it.**

To predict what comes next, a model needs more than the latest token in isolation; it needs the context before it. In "The cat sat down because it was tired," the information associated with "cat" is what makes "it" meaningful. Attention gives the model a way to compare the current token against earlier ones and blend in whatever matters for this step.

This happens at every layer, not just for obvious references like pronouns, and different attention heads learn to emphasize different relationships, such as nearby syntax, long-range references, and formatting patterns.

Attention is the reason a KV cache exists at all. Because each token must look back over everything before it, the model needs the earlier tokens' intermediate state available at every step, which makes that state worth keeping.

<!--
slug: query-key-value
question: What are Query, Key, and Value in attention?
aliases: [QKV, Q K V, query key value]
related: [attention, kv-cache, attention-head, grouped-query-attention]
tier: 1
publish: yes
-->
### Query, Key, and Value (QKV)

**Query, Key, and Value are three vectors derived from each token's current representation, and together they work like a lookup.** Each token asks a question, offers a label to be matched against, and holds the content it hands over if it gets picked.

- Query — what this token is looking for.
- Key — what this token offers, for other tokens comparing against it.
- Value — the information this token passes along if it is attended to.

To decide how much "it" should attend to "cat," the model compares "it"'s Query against "cat"'s Key. Repeated against every earlier token, that produces a set of attention weights, and the token's output is built by blending the earlier tokens' Values in proportion to those weights.

The three vectors come from learned numerical operations, so the question-and-label description is an analogy. It is a useful analogy because it matches how each vector is used.

Only two of the three get cached. A Query is used once, at the moment its own token is processed, and can then be discarded. Every later token keeps needing the earlier Keys and Values, which is why the cache is called a **KV** cache.

<!--
slug: attention-head
question: What is an attention head?
aliases: [attention heads, multi-head attention, MHA]
related: [attention, query-key-value, grouped-query-attention]
tier: 1
publish: yes
-->
### Attention head

**An attention head is one parallel instance of attention within a layer, projecting each token into its own smaller query/key/value space.** A layer runs many of them in parallel, which is called multi-head attention.

Splitting the work this way lets different heads specialize: one may track syntactic neighbors, another long-range references, another document structure. In many models the heads together are exactly as wide as the model's hidden size, so splitting into heads divides the existing width instead of adding to it. Llama 3 8B, for example, has 32 heads of 128 dimensions each, which equals its hidden size of 4,096.

Head counts matter for cache sizing. In standard multi-head and grouped-query attention, the number of *key/value* heads, not query heads, is what multiplies into KV cache size.

<!--
slug: grouped-query-attention
question: What is grouped-query attention (GQA)?
aliases: [GQA, grouped query attention]
related: [attention-head, kv-cache, query-key-value]
tier: 1
publish: yes
-->
### Grouped-query attention (GQA)

**Grouped-query attention is an architecture choice where several Query heads share a single Key/Value head**, rather than each Query head having its own.

In standard multi-head attention the counts are equal. Under GQA there are fewer K/V heads than Q heads, which directly reduces how much key/value data has to be stored per token and read on every decode step.

This is one of the largest single factors in KV cache footprint, and the model's designers choose it, not the serving stack. Llama 3 8B, for example, has 32 Query heads and 8 Key/Value heads, so it stores one quarter of the key/value data that the same model would store with one Key/Value head per Query head.

<!--
slug: causal-mask
question: What is a causal mask?
aliases: [causal masking, causal attention]
related: [attention, prefill, autoregressive-generation, kv-cache]
tier: 1
publish: yes
-->
### Causal mask

**A causal mask blocks a token from attending to positions that come after it**, so each token can use itself and the tokens before it, but nothing ahead.

This one-way flow is what makes caching possible. Because a later token cannot change what an earlier token saw, extending a prefix with a new token does not invalidate the Keys and Values already computed for the earlier positions. The new token attends to them; the model never has to build them again.

The mask also lets prefill process an entire prompt in one parallel pass, because no position can see the tokens that come after it.

<!--
slug: autoregressive-generation
question: What is autoregressive generation?
aliases: [autoregressive decoding, autoregression]
related: [decode, causal-mask, kv-cache, inter-token-latency]
tier: 1
publish: yes
-->
### Autoregressive generation

**Autoregressive generation is the loop a model runs to produce text: generate one token, append it to the sequence, then use the longer sequence to generate the next.**

Every pass through the loop reads the Keys and Values for the entire context built so far, but computes new Keys and Values only for the one new token. Nothing earlier is recomputed, as long as the cache still holds the earlier Keys and Values.

This loop is why each output token takes so much more time than each input token. Prompt tokens can be processed together in one parallel pass. In standard decoding, output tokens are produced one at a time, with one model invocation for each token. Techniques such as speculative decoding can produce several tokens per invocation of the main model, but the underlying loop is still sequential.

<!--
slug: context-length
question: What is context length?
aliases: [context window, max model length, max_model_len]
related: [token, kv-cache, prefill, quantization]
tier: 1
publish: yes
-->
### Context length

**Context length is the maximum number of tokens a model can attend over in a single request** — prompt plus generated output together.

In a serving deployment, the configured context length also affects memory. In vLLM the setting is `--max-model-len`, and the engine refuses to start if its KV cache memory cannot hold at least one request of that full length. vLLM does not reserve that much memory for every request; it allocates cache blocks as each request grows. Lowering the limit is one of the settings vLLM's documentation suggests when GPU memory is short.

Long contexts are where cache reuse saves the most computation, and they are also the source of the memory pressure that makes keeping a cache difficult.

---
## Tier 2: Prefill, decode, and the KV cache

<!--
slug: prefill
question: What is prefill in LLM inference?
aliases: [prefill phase, prompt processing]
related: [decode, kv-cache, ttft, prefill-throughput]
tier: 2
publish: yes
-->
### Prefill

**Prefill is the phase where the model processes the prompt, before it generates anything.**

Because every prompt token is already known, their computations can run in parallel within each layer: one matrix multiplication produces every token's Query, Key, and Value at once, and another produces every position's attention scores against every other position, in a single pass. The [causal mask](/glossary/causal-mask) still prevents any position from using future context.

As the prompt moves through the model, the engine produces and stores a Key and Value for every prompt token at every layer. That is the initial KV cache — the model's understanding of everything in the prompt, saved for reuse.

For prompts that are not very short, prefill is compute-bound, and it is the main compute cost in [time to first token](/glossary/ttft). Cache reuse skips the part of prefill that a cached prefix covers; the uncached remainder still has to be processed.

<!--
slug: decode
question: What is decode in LLM inference?
aliases: [decode phase, generation phase]
related: [prefill, autoregressive-generation, inter-token-latency, decode-throughput]
tier: 2
publish: yes
-->
### Decode

**Decode is the phase where the model generates output, one token at a time.**

For each new token, every layer computes the Query, Key, and Value for that one position only, which is a much smaller computation than the whole-prompt pass of prefill. The Query attends to the Keys already stored for that layer, uses the corresponding Values, and continues through the rest of the model. The new Key and Value are then appended to the cache and the loop repeats.

A single decode step is memory-bandwidth-bound rather than compute-bound, because it reads the model weights and the cached Keys and Values from GPU memory but does little arithmetic for each byte it reads. Engines therefore batch decode steps from many requests together. This hardware profile differs from that of prefill, which is why some deployments separate the two phases ([P/D disaggregation](/glossary/pd-disaggregation)).

Caching does not make every decode step cost the same. As the sequence grows, the new Query has more cached Keys and Values to read and attend over. What caching removes is re-running the model for old positions to recreate state the engine already computed.

<!--
slug: kv-cache
question: What is a KV cache?
aliases: [KV caching, key-value cache, KV store]
related: [prefill, decode, query-key-value, prefix-caching, cache-hit-rate]
tier: 2
publish: yes
-->
### KV cache

**A KV cache stores the Key and Value tensors a model has already computed for the tokens it has seen, so that those tokens do not have to be processed again.** It spans every cached token position at every transformer layer.

You can think of it as the model's working memory of an exchange. Within one sequence, it grows by one entry per layer for every token the model processes.

Without it, producing the 100th token means recomputing Keys and Values for all 99 before it, and the cost compounds: total work across a reply grows roughly with the square of its length. A 100-token reply costs on the order of 5,000 recomputations; a 1,000-token reply costs roughly 500,000 — ten times the length, about a hundred times the work. With a cache, each step computes exactly one new Key and Value per layer, so the number of Key and Value computations grows linearly with length. Each step still reads every cached entry to compute attention, as described under [decode](/glossary/decode).

The tradeoff is memory. Cache footprint scales with cached tokens, layers, KV heads, head dimension, and bytes per value, and it grows separately for every active sequence whose state cannot be shared:

```
2 × layers × tokens × kv_heads × head_dim × bytes_per_value
```

For Llama 3 8B at BF16, that works out to **128 KiB per token per sequence** — about 12.5 MiB for 100 tokens, and roughly 1 GiB for 8,192 tokens, before allocator overhead. That memory competes directly with model weights, activations, and everything else the GPU has to hold, which is why deciding where to keep the cache is a real systems problem.

<!--
slug: prefix-caching
question: What is prefix caching?
aliases: [prefix cache, prefix reuse, cross-request caching]
related: [kv-cache, cache-hit-rate, cache-aware-routing, cacheblend]
tier: 2
publish: yes
-->
### Prefix caching

**Prefix caching reuses the KV cache of an earlier request when a new request starts with the same tokens, so that the shared beginning skips prefill.**

Inside a single generation, reuse is simply the normal decode path. Reuse between requests is a separate problem, because a chat interface shows one continuous conversation but the backend usually receives each turn as a fresh request containing the whole history so far. If the serving system still holds the matching prefix — and can recognize and reach it — it can skip prefill for those tokens and process only the new suffix. If the cache was discarded, evicted, or left on a different engine, the entire history is processed again.

The workloads where this matters most:

- Chat requests that repeatedly carry the same system prompt and conversation history.
- Agents making a series of calls that each include the earlier instructions, tool results, and intermediate state.
- RAG applications reusing the same retrieved documents across related requests.
- Many users or requests sharing an identical opening prefix.

<!--
slug: cache-hit
question: What is a cache hit or cache miss?
aliases: [cache miss, hit, miss]
related: [cache-hit-rate, prefix-caching, chunk-size, cacheblend]
tier: 2
publish: yes
-->
### Cache hit / cache miss

**A cache hit means the Key and Value tensors for a stretch of tokens are already stored and can be reused; a cache miss means they are not, so that part of prefill must run.**

Ordinary prefix caching requires an *exact* prefix match. If one earlier token differs, everything after it misses, and because matching works on whole blocks or [chunks](/glossary/chunk-size), the chunk that contains the difference misses as well. That strictness is what [CacheBlend](/glossary/cacheblend) tries to relax.

A miss costs about what the same request would cost with no cache at all, plus the time spent on the lookup. A benchmark that does not disclose its hit rate is therefore reporting a number whose most important variable is hidden.

<!--
slug: chunk-size
question: What is chunk size in LMCache?
aliases: [chunk_size, cache granularity]
related: [cache-hit-rate, kv-cache, eviction-policy, cacheblend]
tier: 2
publish: yes
-->
### Chunk size

**Chunk size is the granularity at which cached KV data is stored and looked up** — 256 tokens by default in LMCache (the server's `--chunk-size` setting). The setting is a minimum: when vLLM connects, the connector can raise it so that chunk boundaries line up with the model's KV cache layout.

It is a tradeoff rather than a value to maximize. Smaller chunks match more of a partially shared prefix, which raises hit rate on workloads whose shared text does not end on a chunk boundary, but they add per-chunk lookup and metadata overhead. Larger chunks are cheaper to manage and transfer, but a token that differs anywhere inside a chunk makes the whole chunk unusable, so a larger chunk loses more of the matching tokens that come before the difference.

<!--
slug: eviction-policy
question: What is a cache eviction policy?
aliases: [eviction, cache_policy, LRU]
related: [kv-cache, cache-hit-rate, cache-tiering, chunk-size]
tier: 2
publish: yes
-->
### Eviction policy

**An eviction policy decides what to discard when a cache pool is full.** In LMCache's multiprocess server, the policy is a required setting (`--eviction-policy`: `LRU`, `ARC`, `IsolatedLRU`, or `noop`), and the documented examples use LRU, which discards the least recently used entry first. The older in-process LMCache, now maintained as a separate package, set this with `cache_policy` and defaulted to LRU.

Eviction is where a cache stops being a pure win. Once a useful cache is gone and no copy exists elsewhere, the next matching request pays full prefill again. Policy choice, pool size, and workload shape jointly determine how often that happens, which makes eviction one of the direct levers on [cache hit rate](/glossary/cache-hit-rate).

<!--
slug: cache-tiering
question: What is KV cache offloading or tiering?
aliases: [KV offloading, cache tiers, tiered storage, L1 L2 L3 cache]
related: [kv-cache, interconnects, ttft, total-cost-of-ownership]
tier: 2
publish: yes
-->
### Cache tiering and offloading

**Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower storage, so that it stays reusable instead of being discarded.**

In LMCache's multiprocess mode, there are two tiers below GPU memory. L1 is CPU memory by default, and its size (`--l1-size-gb`) has no default, so it must be set. L2 is one or more secondary storage backends, such as a local file system or an S3-compatible object store, configured with `--l2-adapter`. LMCache copies KV data from L1 to L2 in the background and loads it back into L1 when a request needs it.

The bet is economic — cheap storage substituting for expensive GPU memory and compute. It is worth it only when retrieval is faster than recomputation, which depends on how much context is reused, where the cache landed, transfer bandwidth, and whether the lookup hits at all.

The systems questions are therefore what to keep, where to keep it, when to move it, and how to make it reachable by the engine that needs it.

<!--
slug: cacheblend
question: What is CacheBlend?
aliases: [cache blending, enable_blending]
related: [prefix-caching, cache-hit, kv-cache, ttft]
tier: 2
publish: yes
-->
### CacheBlend

**CacheBlend is a technique, implemented in LMCache, for reusing the cached KV data of a text chunk even when the chunk is not at the start of the prompt** — most usefully in RAG workloads, where independently retrieved document chunks get concatenated in whatever order retrieval returned them. It was introduced in the CacheBlend paper (Yao et al., EuroSys 2025).

The problem it addresses: in a causal transformer, every token's Key and Value are built from that token's representation *after attending to everything before it*. A chunk cached on its own never attended to the text that precedes it in the new prompt, so its cached Keys and Values lack that cross-attention and differ from what a full prefill would produce. The chunk also sits at a new position, so its cached Keys must be re-rotated to that position (RoPE). Reusing the cached data without further work therefore gives only an approximate result.

Instead of recomputing the whole chunk, CacheBlend runs every token through the layers up to an early check layer. At that layer it compares each token's freshly computed Key with its cached Key and selects the tokens whose Keys deviate most. Only those tokens, about 15 percent of the reused tokens by default, are recomputed through the remaining layers; the other tokens keep their cached Keys and Values. LMCache exposes this as a single check layer (`cb.check_layer`, default 1) and a recompute ratio (`cb.recomp_ratio`, default 0.15). The paper narrows the selected set gradually over several layers instead.

With an early check layer on a 32-layer model, this is roughly 18–20 percent of the layer computation of a full prefill, and the fraction falls as models get deeper. Loading the cached data from storage is a separate cost, which the paper overlaps with the recomputation. CacheBlend trades a small amount of accuracy for speed, on the assumption that the tokens whose Keys deviate most at the check layer are the ones whose errors matter most. In the paper's evaluation, recomputing 5–18 percent of tokens reduced F1 or Rouge-L by at most 0.002.

> **Status note:** In LMCache's multiprocess server, CacheBlend is off by default; you enable it with `lmcache server --engine-type blend`, and it requires a chunk size of 256. The older in-process implementation (`enable_blending`) supported only Llama, Qwen2, and Qwen3 model architectures. Verify current model support against the release before publishing.

<!--
slug: pd-disaggregation
question: What is prefill-decode (P/D) disaggregation?
aliases: [P/D disaggregation, prefill decode disaggregation, disaggregated serving]
related: [prefill, decode, cache-tiering, cache-aware-routing]
tier: 2
publish: yes
-->
### Prefill–decode (P/D) disaggregation

**P/D disaggregation is running prefill and decode on separate GPU pools or machines instead of both phases on the same worker.**

Prefill is compute-bound and decode is memory-bandwidth-bound, so giving each phase its own pool lets each one run closer to its own hardware limit instead of competing for the same GPUs.

The cost is that the KV cache computed during prefill must be transferred to wherever decode runs. Cache transfer speed therefore becomes part of the latency of every request, instead of an occasional fallback to a remote storage tier.

<!--
slug: cache-aware-routing
question: What is cache-aware routing?
aliases: [prefix-aware routing, cache locality routing]
related: [prefix-caching, cache-hit-rate, load-balancer, pd-disaggregation]
tier: 2
publish: yes
-->
### Cache-aware routing

**Cache-aware routing sends requests likely to share a prefix to the same replica, so the cached data they need is actually reachable.**

In any deployment with more than one replica, the replica that receives a request determines whether a relevant cached prefix exists locally at all. A request routed to a replica that does not hold the matching cache misses on that replica, however well its cache is configured, unless the deployment shares cache data between replicas (for example, through a shared storage tier or peer-to-peer transfer).

A standard Kubernetes Service picks a backend Pod at random by default and does not inspect request content, so it has no concept of cache locality. Routing is therefore a separate lever on effective hit rate, and a multi-replica benchmark should state which routing policy it used.

<!--
slug: lmcache
question: What is LMCache?
aliases: [LM Cache]
related: [kv-cache, cache-tiering, prefix-caching, pd-disaggregation]
tier: 2
publish: yes
-->
### LMCache

**LMCache is an open source KV cache layer for LLM serving.** It treats the KV cache as reusable data that can live beyond a single GPU-memory allocation and, when useful, beyond a single request or inference-engine process.

It moves KV data through a storage hierarchy spanning GPU memory, CPU memory, local storage, and remote backends, so that when a later request reuses cached context the serving system can retrieve the corresponding state rather than rebuilding it through prefill. That can cut repeated computation, improve time to first token, and leave GPU memory free for active work.

LMCache integrates with the inference engine rather than replacing it — most commonly vLLM, through vLLM's KV connector interface. It also includes integrations for SGLang and TensorRT-LLM.

---
## Tier 3: Performance metrics

<!--
slug: ttft
question: What is time to first token (TTFT)?
aliases: [TTFT, time to first token]
related: [prefill, tail-latency, inter-token-latency, cache-hit-rate]
tier: 3
publish: yes
-->
### Time to first token (TTFT)

**TTFT is the delay between a request arriving and the model producing its first output token.** When requests are not waiting in a queue, most of that delay is prefill, and prefill is the work that KV cache reuse removes most directly: a cache hit skips re-processing context the system has already seen.

One caveat changes how the number should be read. TTFT is measured from request *arrival*, so it includes time spent queued before the server started work at all. A benchmark that does not report the load level relative to server capacity can hide a large queueing contribution inside what looks like a pure compute number.

<!--
slug: inter-token-latency
question: What is inter-token latency (ITL) or time per output token (TPOT)?
aliases: [ITL, TPOT, time per output token, inter token latency]
related: [decode, interactivity, ttft, decode-throughput]
tier: 3
publish: yes
-->
### Inter-token latency (ITL / TPOT)

**ITL is the time between two consecutive output tokens once generation has started.** Time per output token (TPOT) is the same quantity averaged over one request, and many tools use the two names interchangeably. vLLM reports them as separate metrics, however: ITL as the distribution of every gap between tokens, and TPOT as one average per request.

Where TTFT governs how long a user waits before anything appears, ITL governs how smoothly the response streams once it starts.

<!--
slug: interactivity
question: What is interactivity in inference benchmarks?
aliases: [tokens per second per user, tok/s/user]
related: [inter-token-latency, decode-throughput, concurrency]
tier: 3
publish: yes
-->
### Interactivity

**Interactivity is tokens delivered per second per user** — roughly the inverse of inter-token latency.

It is the same underlying phenomenon as ITL, expressed as a rate rather than a delay. Some benchmarks plot it instead of ITL because "higher is better" reads more intuitively than "lower is better." Check which one a chart shows before you compare results across sources.

<!--
slug: tail-latency
question: What is tail latency (p50, p90, p99)?
aliases: [p99, p90, percentile latency, tail]
related: [ttft, inter-token-latency, concurrency]
tier: 3
publish: yes
-->
### Tail latency (p50 / p90 / p99)

**Tail latency is the latency of the slowest requests, reported as high percentiles such as p90 or p99 rather than as an average.** p99 means 99 percent of requests were at least this fast, and 1 percent were slower. p50, the median, describes the typical request and is the usual point of comparison.

Real request latencies are right-skewed: most are fast, but a subset run much slower because of cache misses, resource contention, eviction, or retries, stretching a long tail to the right. A mean barely moves in response to that tail. Percentiles look directly at where it sits.

This matters because the user who landed in the slow 1 percent had a genuinely bad experience regardless of how good the average looked. Production service-level objectives (SLOs) are usually written against tail latency for this reason, which means a benchmark reporting only mean TTFT can look substantially better than what real users experience.

<!--
slug: decode-throughput
question: What is decode throughput?
aliases: [output token throughput, generation throughput]
related: [prefill-throughput, qps, concurrency, interactivity]
tier: 3
publish: yes
-->
### Decode throughput

**Decode throughput is the aggregate rate of output tokens per second across all concurrent requests.**

It measures steady-state generation capacity, which is a different axis from latency — and the two trade off against each other as concurrency changes.

<!--
slug: prefill-throughput
question: What is prefill throughput?
aliases: [prompt processing throughput]
related: [prefill, ttft, decode-throughput]
tier: 3
publish: yes
-->
### Prefill throughput

**Prefill throughput is the aggregate rate at which the system processes prompt tokens, in tokens per second.** It is the throughput-side counterpart to TTFT: same underlying work, measured as capacity rather than as delay.

<!--
slug: qps
question: What is QPS in inference serving?
aliases: [queries per second, requests per second, RPS]
related: [decode-throughput, concurrency, tail-latency]
tier: 3
publish: yes
-->
### QPS (queries per second)

**QPS (queries per second) is a request rate; in LLM serving, one query is one request, such as a single chat-completion call.**

Benchmarks use the name for two different numbers, so check which one a result reports. The first is the rate at which the benchmark *sends* requests, which sets the offered load; vLLM's `--request-rate` and the `--mrc-qps` option of `lmcache bench engine` work this way. The second is the rate at which requests *complete*, often reported as requests per second or request throughput. While the server keeps up, the two rates are equal. When it cannot keep up, requests wait in a queue, latency grows, and the completed rate stays below the send rate.

Requests differ in prompt length and output length, so a QPS figure is comparable only between benchmarks that use the same workload.

<!--
slug: cache-hit-rate
question: What is cache hit rate?
aliases: [hit rate, KV cache hit rate, prefix cache hit rate]
related: [cache-hit, prefix-caching, eviction-policy, chunk-size]
tier: 3
publish: yes
-->
### Cache hit rate

**Cache hit rate is the share of requested tokens that were served from cache rather than recomputed.** Every latency or throughput gain claimed for a KV cache depends on it.

KV caching only helps when parts of a prompt repeat across requests — a shared system prompt, a repeated document, an earlier turn in the same conversation. If none of a request's tokens are in the cache, caching does not reduce that request's prefill work at all.

Engines that reuse only a matching prefix report this number as the *prefix cache hit rate*. vLLM, for example, exposes the counters `vllm:prefix_cache_hits` and `vllm:prefix_cache_queries`, and the hit rate is their ratio. LMCache's token-level hit rate also counts only the matching prefix, in units of whole chunks. CacheBlend is different, because it can also reuse cached chunks that are not part of a prefix; LMCache therefore reports blend hits with their own metrics.

The saved prefill work grows with the hit rate. A workload with heavy prompt overlap saves far more total TTFT and compute than a workload of unique one-time prompts, where the hit rate is near zero.

Hit rate depends on the workload's actual repetition pattern and also on cache pool size, chunk size, and eviction policy. A benchmark that neither discloses nor controls the hit rate therefore reports a number that nobody can reproduce.

<!--
slug: concurrency
question: How do concurrency and batching affect inference performance?
aliases: [batching, batch size, continuous batching]
related: [decode-throughput, interactivity, tail-latency, qps]
tier: 3
publish: yes
-->
### Concurrency and batching

**Concurrency is how many requests a server works on at once; batching is how it groups them so the GPU processes them together.**

LLM servers such as vLLM and SGLang use *continuous batching*, which the Orca paper introduced as iteration-level scheduling. The server rebuilds the batch at every generation step instead of once per batch. As a result, a request that finishes leaves the batch immediately, and a waiting request can join at the next step instead of waiting for the whole batch to finish.

Concurrency sets the trade-off between the two headline metrics. As concurrency rises, total decode throughput rises until the GPU's compute is saturated, but each user receives tokens more slowly (lower interactivity), and TTFT grows because new requests share the GPU with more running requests.

For this reason, a result that reports only throughput or only latency, without the concurrency level it was measured at, cannot be compared with a result measured at a different concurrency level, even on identical hardware.

<!--
slug: total-cost-of-ownership
question: What is TCO for LLM inference?
aliases: [TCO, total cost of ownership, cost per GPU hour, cost per GB]
related: [cache-tiering, qps, decode-throughput]
tier: 3
publish: yes
-->
### Total cost of ownership (TCO)

**TCO is the cost of running a deployment over its lifetime, not its sticker price.** In caching discussions it usually shows up as a comparison between two units: dollars per GB of tiered storage against dollars per GPU-hour.

That comparison is the bet a caching layer makes: cheaper storage substitutes for more expensive GPU memory and compute.

> **Sourcing note:** public numbers in this space are overwhelmingly vendor-sourced, including ours. Any figure we publish should be labeled as such, and independent GPU and storage pricing is worth gathering before we put dollar amounts in front of readers.

---

## Tier 4: Engine and hardware knobs

<!--
slug: quantization
question: What is quantization in LLM inference?
aliases: [precision, dtype, fp8, kv cache dtype]
related: [model-weights, kv-cache, cache-hit-rate]
tier: 4
publish: yes
-->
### Quantization and precision

**Quantization is storing and computing numbers at lower precision to save memory and time.** Two separate settings are often confused:

- Weight precision: how the model's own parameters are stored and computed. In vLLM, `--dtype` selects only 16- or 32-bit formats (fp16, bf16, fp32). Lower-precision weights such as fp8 or fp4 come from a quantized model checkpoint or from the `--quantization` option. On hardware with dedicated support, lower precision makes matrix multiplication faster, which speeds up both prefill and decode regardless of caching.
- KV cache precision (`--kv-cache-dtype`, a separate flag): how the cached key/value tensors are stored, independent of weight precision. A lower-precision KV cache takes less memory per token, so more tokens fit in the same pool, which raises the ceiling on hit rate. Depending on the attention backend, the kernel either converts the cached values back to higher precision when it reads them or runs attention directly in fp8.

Precision is also hardware-dependent. vLLM runs fp8 computation only on NVIDIA GPUs with compute capability 8.9 or higher (Ada Lovelace, Hopper, Blackwell). On older GPUs such as Turing and Ampere, it runs an fp8 model as weight-only fp8: the weights are stored in fp8, but the arithmetic runs at 16-bit precision, so the compute speedup of fp8 is not available. The same setting therefore behaves differently on different hardware.

Lower precision is an approximation, so it can cost output quality. Below a certain precision, which depends on the model, the quality loss becomes measurable.

<!--
slug: tensor-parallelism
question: What is tensor parallelism?
aliases: [TP, tensor-parallel-size]
related: [pipeline-parallelism, expert-parallelism, interconnects, kv-cache]
tier: 4
publish: yes
-->
### Tensor parallelism (TP)

**Tensor parallelism splits each layer's weight matrix across several GPUs**, so each GPU computes part of one matrix multiplication and the GPUs then exchange and combine their partial results before moving to the next layer.

Because that combining step happens at *every single layer*, TP needs a very fast GPU-to-GPU link, such as NVLink rather than PCIe, to avoid being dominated by communication. For this reason, vLLM's documentation recommends TP across the GPUs inside one node and pipeline parallelism across nodes, and it recommends pipeline parallelism instead of TP when the GPUs in a node have no NVLink.

On a slow interconnect, adding more TP can therefore make inter-token latency worse rather than better.

<!--
slug: pipeline-parallelism
question: What is pipeline parallelism?
aliases: [PP]
related: [tensor-parallelism, expert-parallelism, interconnects]
tier: 4
publish: yes
-->
### Pipeline parallelism (PP)

**Pipeline parallelism assigns different whole layers to different GPUs**, so a token's forward pass flows through them like an assembly line — GPU 0 handles layers 1–20, GPU 1 handles 21–40.

Communication happens only at stage handoffs, far less than TP requires, so PP tolerates slower interconnects and can span separate machines. The cost is pipeline *bubbles* — idle GPU time — unless enough requests are kept in flight to keep every stage busy.

<!--
slug: expert-parallelism
question: What is expert parallelism?
aliases: [EP, MoE parallelism]
related: [tensor-parallelism, pipeline-parallelism]
tier: 4
publish: yes
-->
### Expert parallelism (EP)

**Expert parallelism places different experts of a mixture-of-experts model on different GPUs**, routing each token to the GPUs that hold the experts its router selected.

The communication pattern is an all-to-all exchange: each token is sent out to the GPUs that hold its selected experts, and the results are sent back. This pattern differs in shape from TP's per-layer combine, and it applies only to MoE architectures.

EP frees memory indirectly: each GPU holds only a fraction of the expert weights, which leaves more room for KV cache. EP does not shard attention, however, so how KV data is distributed depends on how the attention layers are parallelized (data, tensor, or pipeline parallel), not on EP itself.

<!--
slug: speculative-decoding
question: What is speculative decoding?
aliases: [draft model, speculative sampling]
related: [decode, inter-token-latency, decode-throughput]
tier: 4
publish: yes
-->
### Speculative decoding

**Speculative decoding cheaply proposes several tokens ahead, often with a smaller *draft* model, and the main model then verifies them in a single pass.**

Other proposal methods exist besides a draft model; vLLM, for example, also supports EAGLE, multi-token prediction (MTP), and n-gram matching. Speculative decoding reduces inter-token latency mainly when the server is under low to medium load, and its gains shrink as load rises. It works separately from KV cache reuse, and the two can run together. It is listed here because it moves the same metrics that caching does, which means a speedup in a benchmark is not always caused by caching. LMCache does not implement speculative decoding.

<!--
slug: attention-kernel
question: What is an attention kernel?
aliases: [kernel, FlashAttention, attention implementation]
related: [attention, decode, inter-token-latency]
tier: 4
publish: yes
-->
### Attention kernel

**A kernel is a low-level GPU program implementing one specific computation as hardware-optimized code; the attention kernel is the code that actually runs attention's matrix multiplications.**

Two engines — or two versions of the same engine — running an identical model on identical hardware can differ in speed purely because of which attention kernel they use. The math is the same, but different code runs at different speeds.

This is why a benchmark comparing engine versions is partly a kernel-implementation comparison, not purely a comparison of whatever feature it claims to be testing.

<!--
slug: interconnects
question: What are PCIe, NVLink, and RDMA?
aliases: [PCIe, NVLink, RDMA, interconnect, fabric]
related: [cache-tiering, tensor-parallelism, pd-disaggregation, ttft]
tier: 4
publish: yes
-->
### Interconnects (PCIe, NVLink, RDMA)

**Interconnects are the links data travels over between GPUs, machines, and storage.** The interconnect limits how fast cached KV data can move, so a benchmark result depends on which one the data crossed.

- PCIe: the standard bus connecting a GPU to the rest of its machine (CPU, RAM). Speed depends on generation and lane count.
- NVLink: a much faster GPU-to-GPU interconnect than PCIe, available only on hardware that supports it. On some systems, such as NVIDIA's Grace-based superchips, a related link (NVLink-C2C) also connects the GPU to the CPU.
- RDMA: lets one machine read and write another machine's memory directly over the network, bypassing the remote CPU. Much faster and lower-latency than a conventional TCP/HTTP path. LMCache's peer-to-peer (P2P) KV cache sharing uses one-sided RDMA reads to pull cached data from another node's memory.

Any claim that cache retrieval is faster than recomputation is, underneath, a claim about which of these links the data crossed.

---
## Tier 5: Deployment and systems

*These are support terms. They exist so the documents above do not have to re-explain infrastructure vocabulary inline. Most are better served as short entries on the glossary index than as standalone pages — they are not terms anyone should land on lmcache.ai to learn.*

<!--
slug: process
question: What is a process?
aliases: [OS process]
related: [thread, container, pod]
tier: 5
publish: support
-->
### Process

**A process is one running instance of a program**, with its own private memory. Processes exchange data only through explicit mechanisms such as network connections, shared memory, or message queues.

Launch the same script in two terminal windows and you have two separate processes. A single vLLM plus LMCache deployment on one machine typically runs several at once. For example, vLLM starts one worker process per GPU, so `--tensor-parallel-size 4` means four separate GPU worker processes.

<!--
slug: thread
question: What is a thread?
aliases: [threads, threading]
related: [process, container]
tier: 5
publish: support
-->
### Thread

**A thread is the thing that actually executes instructions inside a process.** Every process starts with one; a process can ask the operating system for more, and they all share the same process memory.

The reason to want several: a process often needs to do things that should not block each other. In LMCache, one thread accepts incoming lookup and store requests, another handles the completion signals from GPU memory copies, and another periodically evicts old entries. On a single thread, one slow step would freeze all the others.

If a process is an office building with one shared set of filing cabinets, threads are the employees — each doing different work, all reading and writing the same files. That sharing is why they are fast and also why they need coordination.

<!--
slug: container
question: What is a container?
aliases: [containers, Docker container]
related: [pod, node, process]
tier: 5
publish: support
-->
### Container

**A container is an isolated process packaged with everything its application needs to run** — code, dependencies, system libraries — so it behaves consistently on different machines.

A container is lighter than a virtual machine because it shares the host's operating system kernel instead of running a whole separate operating system with its own kernel. As a result, many containers can run on one machine with little overhead. The shared kernel also means a container still depends on the host: a GPU container, for example, uses the GPU driver installed on the host. The standard analogy is a shipping container, whose contents stay the same no matter which ship or truck carries it.

<!--
slug: node
question: What is a node in Kubernetes?
aliases: [nodes]
related: [pod, cluster, container]
tier: 5
publish: support
-->
### Node

**A node is one physical or virtual machine that is part of a cluster**, capable of running some number of pods.

<!--
slug: pod
question: What is a Kubernetes pod?
aliases: [pods]
related: [node, cluster, replica, container]
tier: 5
publish: support
-->
### Pod

**A pod is Kubernetes' smallest deployable unit** — usually one container, occasionally a small group of tightly-coupled containers, scheduled onto a node together.

When people casually say "an instance" or "a replica" of a service, they usually mean one pod.

<!--
slug: cluster
question: What is a Kubernetes cluster?
aliases: [clusters, fleet]
related: [node, pod, replica]
tier: 5
publish: support
-->
### Cluster

**A cluster is a set of nodes plus the control plane that manages them as one system.**

*Fleet* is the informal version of the same idea — the group of running replicas considered together, without implying the specific Kubernetes machinery underneath.

<!--
slug: replica
question: What is a replica in Kubernetes?
aliases: [replicas, Deployment, ReplicaSet, autoscaling]
related: [pod, cluster, load-balancer, cache-aware-routing]
tier: 5
publish: support
-->
### Replica (Deployment / ReplicaSet)

**A replica is one running copy of a service.** In Kubernetes, a ReplicaSet keeps a specified number of identical pods running, and a Deployment manages ReplicaSets; the Deployment is usually where you declare "keep N copies of this pod running."

Autoscaling means adjusting that N automatically as load changes. The Kubernetes HorizontalPodAutoscaler, for example, updates a Deployment's replica count to match demand.

Replica count is where cache locality becomes a problem: more replicas means more places a cached prefix might be, and only one of them is where the next request will land.

<!--
slug: load-balancer
question: What is a Kubernetes Service or load balancer?
aliases: [Service, load balancing, routing layer]
related: [replica, cache-aware-routing, cluster]
tier: 5
publish: support
-->
### Service / load balancer

**A Service is a stable network entry point that spreads incoming connections across the ready pods behind a deployment.** It is the routing layer, which decides which replica a given request actually reaches.

Plain Kubernetes load balancing has no concept of cache locality. It has no information about which replica might already hold relevant cached data, and [cache-aware routing](/glossary/cache-aware-routing) exists to fill that gap.

<!--
slug: operator
question: What is a Kubernetes Operator?
aliases: [operators, custom controller]
related: [cluster, replica, liveness-probe]
tier: 5
publish: support
-->
### Kubernetes Operator

**An Operator is a custom controller that knows how to manage one particular piece of software's lifecycle** — an automation pattern built on top of plain Kubernetes.

Where Kubernetes generically keeps N pods running, an Operator understands application-specific concerns: how to safely scale down a cache server without losing in-flight lookups, how to roll out a new engine version without dropping requests. Operators are optional, and many deployments do not use one.

<!--
slug: liveness-probe
question: What are liveness and readiness probes?
aliases: [readiness probe, health check, healthz]
related: [pod, operator, replica]
tier: 5
publish: support
-->
### Liveness and readiness probes

**Probes are how Kubernetes periodically asks a running container whether it is healthy**, usually by requesting a URL like `/healthz` and checking the response.

When a **liveness** probe keeps failing, Kubernetes assumes the container is stuck and restarts it. When a **readiness** probe fails, Kubernetes stops routing new traffic to the pod but does not restart it; for example, a pod can report "not ready" while it is still loading model weights.

LMCache's multiprocess coordinator (`mp_coordinator`) serves a `/healthz` endpoint, and the LMCache Kubernetes Operator uses that path for the coordinator's liveness and readiness probes.

<!--
slug: rest
question: What is REST?
aliases: [REST API, RESTful]
related: [process, load-balancer]
tier: 5
publish: support
-->
### REST

**REST is a common style for building APIs over HTTP** — clients use standard HTTP methods (`GET`, `POST`) on resources that each have their own URL, so programs can exchange data with ordinary HTTP tools instead of a custom protocol.

Both sides still have to agree on which URLs exist and what data format each one uses.

---
## Open items before publication

1. **Technical review.** Every Tier 1–4 definition should be read by someone on the core team. The CacheBlend, parallelism, and quantization entries are the highest-risk for drift against the current release.
2. **Version-pin the specifics.** The Tier 2 entries now describe the multiprocess server flags (`--l1-size-gb`, `--eviction-policy`, `--l2-adapter`, `--engine-type blend`) and the CacheBlend defaults (`cb.check_layer` 1, `cb.recomp_ratio` 0.15) as of the November 2026 `dev` branch. Recheck them against each release, and prefer wording that does not state a number that will age.
3. **Cost figures.** The TCO entry deliberately carries no dollar amounts. Do not add any before there is a non-vendor source.
4. **Decide the Tier 5 treatment.** These are support definitions, not audience-facing pages. Recommend index-only entries rather than standalone URLs.
5. **Auto-linking.** Once slugs are stable, the first mention of any term in a blog post should link to its page. That internal linking, not the glossary itself, is where most of the value comes from.
