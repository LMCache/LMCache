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

## Tier 1 — Inference primitives

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

Almost everything expensive about running a language model in production is an inference problem, not a training problem — and most of the expense is about memory and data movement rather than raw arithmetic.

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

A token is roughly ¾ of a word, or about four characters. Common words are usually a single token; rarer ones get split into pieces. Each token has an integer ID, which the model uses to look up its starting representation.

Tokens are the unit that nearly every other number in this glossary is denominated in: context length is measured in tokens, throughput is measured in tokens per second, KV-cache size is measured in bytes per token, and pricing is usually per million tokens.

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

It happens before any model computation. The tokenizer is specific to the model — the same sentence produces a different number of tokens under different tokenizers, which is why token counts from one vendor's model do not transfer cleanly to another's.

Tokenization is also where surprises come from in cost and context accounting: unusual formatting, non-English text, and long strings of code or identifiers can consume far more tokens than their character count suggests.

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

The initial embedding is only a starting point. As the vector passes through the model's layers, its representation shifts based on surrounding context — the token "bank" should contribute differently in "river bank" than in "bank account." The mechanism that lets one token take on information from others is [attention](/glossary/attention).

<!--
slug: model-weights
question: What are model weights?
aliases: [weights, parameters, model parameters]
related: [inference, quantization, attention, tensor-parallelism]
tier: 1
publish: yes
-->
### Model weights

**Model weights are the numbers a model learned during training and keeps fixed during inference.** They are the model — an 8-billion-parameter model has 8 billion of them.

The important property for anything cache-related: weights do not grow with the prompt. A layer's projection matrices are the same size whether you send it four tokens or forty thousand. Everything that *does* grow with the prompt — the activations, the keys and values — is request-dependent data, and that is the part that has to be stored, moved, or thrown away.

That split, fixed weights versus request-dependent tensors, is the line the entire KV-caching problem sits on.

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

This happens at every layer, not just for obvious references like pronouns, and different attention heads learn to emphasize different relationships — nearby syntax, long-range references, formatting patterns.

Attention is the reason a KV cache exists at all. Because each token must look back over everything before it, the model needs the earlier tokens' intermediate state available at every step — so it is worth keeping.

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

- **Query** — what this token is looking for.
- **Key** — what this token offers, for other tokens comparing against it.
- **Value** — the information this token passes along if it is attended to.

To decide how much "it" should attend to "cat," the model compares "it"'s Query against "cat"'s Key. Repeated against every earlier token, that produces a set of attention weights, and the token's output is built by blending the earlier tokens' Values in proportion to those weights. Keys decide *how much* to attend; Values are the content that actually gets used.

These are learned numerical operations, not literal questions or stored definitions — but the lookup framing is the right mental model.

Only two of the three get cached. A Query is used once, at the moment its own token is processed, and can be discarded. The Keys and Values are what every later token keeps needing. Hence: **KV** cache.

<!--
slug: attention-head
question: What is an attention head?
aliases: [attention heads, multi-head attention, MHA]
related: [attention, query-key-value, grouped-query-attention]
tier: 1
publish: yes
-->
### Attention head

**An attention head is one independent copy of the attention computation, running on its own slice of each token's representation.** A layer runs many of them in parallel — this is multi-head attention.

Splitting the work this way lets different heads specialize: one may track syntactic neighbors, another long-range references, another document structure. The total amount of data stored per token does not change by splitting into heads; it is the same width, divided up.

Head counts matter for cache sizing. The number of *key/value* heads, not query heads, is what multiplies into KV-cache size.

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

This is one of the largest single levers on KV-cache footprint, and it is decided by the model architect, not by the serving stack. Llama 3 8B, for instance, has 32 Query heads and 8 Key/Value heads.

<!--
slug: causal-mask
question: What is a causal mask?
aliases: [causal masking, causal attention]
related: [attention, prefill, autoregressive-generation, kv-cache]
tier: 1
publish: yes
-->
### Causal mask

**A causal mask blocks a token from attending to positions that come after it**, enforcing that a model generating text left to right can use itself and everything before it, but nothing ahead.

This one-way flow is what makes caching possible. Because a later token cannot change what an earlier token saw, extending a prefix with a new token does not invalidate the Keys and Values already computed for the earlier positions. The new token attends to them; the model never has to build them again.

The mask is also what lets prefill process an entire prompt in one parallel pass without any position peeking ahead.

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

Every pass through the loop reads the entire context built so far, but computes and appends only one new Key and Value. Nothing earlier is redone — provided the cache is there.

This loop is why output length costs so much more than input length. Prompt tokens can be processed together in one parallel pass; output tokens are strictly sequential, one model invocation each.

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

In a serving deployment this is also a memory decision, not just a model capability. The engine reserves cache space against the configured ceiling (`--max-model-len` in vLLM), so setting it higher than a workload actually needs consumes memory that could otherwise hold cached state for more concurrent requests.

Longer contexts are exactly where cache reuse pays off most, and also where the memory pressure that makes reuse hard comes from.

---

## Tier 2 — Prefill, decode, and the KV cache

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

Prefill is compute-bound, and it is what dominates [time to first token](/glossary/ttft). It is also the work that cache reuse skips.

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

For each new token, every layer computes just that token's Query, Key, and Value — a single small computation, not the batched pass prefill used. The Query attends to the Keys already stored for that layer, uses the corresponding Values, and continues through the rest of the model. The new Key and Value are then appended to the cache and the loop repeats.

Decode is memory-bandwidth-bound rather than compute-bound, which is why it has a different hardware profile from prefill — and why some deployments separate the two ([P/D disaggregation](/glossary/pd-disaggregation)).

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

**A KV cache is the stored collection of Key and Value tensors a model has already computed for the tokens it has seen, kept so those tokens never have to be processed again.** It spans every cached token position at every transformer layer.

You can think of it as the model's working memory of an exchange. It only grows as the session continues.

Without it, producing the 100th token means recomputing Keys and Values for all 99 before it, and the cost compounds: total work across a reply grows roughly with the square of its length. A 100-token reply costs on the order of 5,000 recomputations; a 1,000-token reply costs roughly 500,000 — ten times the length, about a hundred times the work. With a cache, each step contributes exactly one new Key and Value, and cost grows linearly.

The tradeoff is memory. Cache footprint scales with cached tokens, layers, KV heads, head dimension, and bytes per value, and it grows separately for every active sequence whose state cannot be shared:

```
2 × layers × tokens × kv_heads × head_dim × bytes_per_value
```

For Llama 3 8B at BF16, that works out to **128 KiB per token per sequence** — about 12.5 MiB for 100 tokens, and roughly 1 GiB for 8,192 tokens, before allocator overhead. That memory competes directly with model weights, activations, and everything else the GPU has to hold, which is what makes cache placement a real systems problem rather than a free win.

<!--
slug: prefix-caching
question: What is prefix caching?
aliases: [prefix cache, prefix reuse, cross-request caching]
related: [kv-cache, cache-hit-rate, cache-aware-routing, cacheblend]
tier: 2
publish: yes
-->
### Prefix caching

**Prefix caching is reusing a KV cache across separate requests, not just within one generation.**

Reusing the cache inside a single generation is just the normal decode path. Reusing it between requests is a different problem. A chat interface shows one continuous conversation, but the backend usually receives each turn as a fresh request containing the whole history so far. If the serving system still holds the matching prefix — and can recognize and reach it — it can skip prefill for those tokens and process only the new suffix. If the cache was discarded, evicted, or left behind on a different engine, the entire history gets processed again.

The workloads where this matters most:

- Chat requests that repeatedly carry the same system prompt and conversation history.
- Agents making a series of calls that carry forward instructions, tool results, and intermediate state.
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

**A cache hit means the key/value tensors for a stretch of tokens were already computed and can be reused; a cache miss means there is nothing to reuse and the model has to run that portion of prefill from scratch**, exactly as if caching were not there at all.

Ordinary prefix caching requires an *exact* prefix match. If one earlier token differs, everything after it misses. That strictness is what [CacheBlend](/glossary/cacheblend) tries to relax.

A miss is not a penalty — it is simply the uncached baseline. But a benchmark that does not disclose its hit rate is reporting a number whose most important variable is hidden.

<!--
slug: chunk-size
question: What is chunk size in LMCache?
aliases: [chunk_size, cache granularity]
related: [cache-hit-rate, kv-cache, eviction-policy, cacheblend]
tier: 2
publish: yes
-->
### Chunk size

**Chunk size is the granularity at which cached KV data is stored and looked up** — 256 tokens by default in LMCache (`chunk_size`).

It is a genuine tradeoff rather than a value to maximize. Smaller chunks match more partial prefixes, raising hit rate on workloads whose overlap does not fall on tidy boundaries, but add per-chunk lookup and metadata overhead. Larger chunks are cheaper to manage and transfer, but a single differing token near a boundary wastes more of the chunk.

<!--
slug: eviction-policy
question: What is a cache eviction policy?
aliases: [eviction, cache_policy, LRU]
related: [kv-cache, cache-hit-rate, cache-tiering, chunk-size]
tier: 2
publish: yes
-->
### Eviction policy

**An eviction policy decides what to discard when a cache pool is full.** LMCache defaults to LRU — least recently used goes first.

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

**Cache tiering is moving KV data out of GPU memory into progressively cheaper, slower storage, so it stays reusable instead of being thrown away.**

The usual hierarchy: GPU memory, then CPU RAM (L1 in LMCache's configuration, `max_local_cpu_size`, 5 GB by default), then local disk (L2, `max_local_disk_size`), then a remote store (L3, `remote_url`).

The bet is economic — cheap storage substituting for expensive GPU memory and compute. It only pays off when retrieval is faster than recomputation, which depends on how much context is reused, where the cache landed, transfer bandwidth, and whether the lookup hits at all. Retrieving a cache is not free and is not always faster than rebuilding it.

That is the real systems question, and it is not "should we cache." It is what to keep, where to keep it, when to move it, and how to make it reachable by the engine that needs it.

<!--
slug: cacheblend
question: What is CacheBlend?
aliases: [cache blending, enable_blending]
related: [prefix-caching, cache-hit, kv-cache, ttft]
tier: 2
publish: yes
-->
### CacheBlend

**CacheBlend is an LMCache technique for reusing cached KV data across chunks that are *not* in the position they were cached in** — most usefully in RAG workloads, where independently retrieved document chunks get concatenated in whatever order retrieval returned them.

The problem it addresses: in a causal transformer, every token's key and value are built from that token's representation *after attending to everything before it*. A chunk cached in isolation never attended to whatever now precedes it once it is spliced into a new prompt, so every one of that chunk's tokens carries a cached key/value reflecting the wrong preceding context. That mismatch is the approximation.

The fix: rather than recompute the whole spliced-in chunk, CacheBlend picks one designated layer, computes the true key there for every token in the combined sequence, ranks all tokens by how far their cached key deviates, and fully recomputes only the worst ~15% through the remaining layers. The other ~85% stay frozen at their cached values.

For a 32-layer model that is roughly 18% of the cost of recomputing from scratch, and the ratio improves on deeper models. It is explicitly a speed-for-accuracy trade: the bet is that the tokens judged to deviate most were the ones that mattered most.

> **Status note:** CacheBlend is off by default (`enable_blending`) and currently supports Llama, Qwen2, and Qwen3 only. Verify against the current release before publishing.

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

The reason: prefill is compute-bound and decode is memory-bandwidth-bound, so dedicating separate pools lets each run closer to its own hardware ceiling rather than sharing one pool awkwardly.

The cost: the KV cache computed during prefill has to physically move to wherever decode happens. That turns cache transfer speed from an occasional remote-tier fallback into a first-class part of the latency story.

A subtlety most explainers miss — caching does not shrink the attention computation itself. It removes the redundant recompute around it.

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

In any deployment with more than one replica, whichever replica a request lands on determines whether a relevant cached prefix exists locally at all. A request routed to a replica that does not hold the matching cache is a guaranteed miss, no matter how well-tuned that replica's cache configuration is.

Plain Kubernetes load balancing has no concept of cache locality. This is why routing is a real, separate lever on effective hit rate, and why it belongs in any honest account of a multi-replica benchmark.

<!--
slug: lmcache
question: What is LMCache?
aliases: [LM Cache]
related: [kv-cache, cache-tiering, prefix-caching, pd-disaggregation]
tier: 2
publish: yes
-->
### LMCache

**LMCache is an open-source KV-cache layer for LLM serving.** It treats the KV cache as reusable data that can live beyond a single GPU-memory allocation and, when useful, beyond a single request or inference-engine process.

It moves KV data through a storage hierarchy spanning GPU memory, CPU memory, local storage, and remote backends, so that when a later request reuses cached context the serving system can retrieve the corresponding state rather than rebuilding it through prefill. That can cut repeated computation, improve time to first token, and leave GPU memory free for active work.

LMCache integrates with the inference engine rather than replacing it — most commonly vLLM, through connector hooks in the engine's own scheduler.

---

## Tier 3 — Performance metrics

<!--
slug: ttft
question: What is time to first token (TTFT)?
aliases: [TTFT, time to first token]
related: [prefill, tail-latency, inter-token-latency, cache-hit-rate]
tier: 3
publish: yes
-->
### Time to first token (TTFT)

**TTFT is the delay between a request arriving and the model producing its first output token.** It is dominated by prefill, and it is the metric KV-cache reuse targets most directly — a hit skips re-processing context the system has already seen.

One caveat that changes how the number should be read: TTFT is measured from request *arrival*, so it includes time spent queued before the server started work at all. A benchmark that does not report the load level relative to server capacity can hide a large queueing contribution inside what looks like a pure compute number.

<!--
slug: inter-token-latency
question: What is inter-token latency (ITL) or time per output token (TPOT)?
aliases: [ITL, TPOT, time per output token, inter token latency]
related: [decode, interactivity, ttft, decode-throughput]
tier: 3
publish: yes
-->
### Inter-token latency (ITL / TPOT)

**ITL is the delay between each subsequent output token once generation has started.** Time per output token (TPOT) is the same measurement under a different name.

Where TTFT governs how long a user waits before anything appears, ITL governs how choppy the response feels once it starts streaming.

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

It is the same underlying phenomenon as ITL, expressed as a rate rather than a delay. Some benchmarks plot it instead of ITL because "higher is better" reads more intuitively than "lower is better." Be aware of which one a chart is showing before comparing across sources.

<!--
slug: tail-latency
question: What is tail latency (p50, p90, p99)?
aliases: [p99, p90, percentile latency, tail]
related: [ttft, inter-token-latency, concurrency]
tier: 3
publish: yes
-->
### Tail latency (p50 / p90 / p99)

**Tail latency is the latency experienced by the slowest requests, reported as percentiles rather than an average.** p99 means 99% of requests were at least this fast, and 1% were slower.

Real request latencies are right-skewed: most are fast, but a subset run much slower because of cache misses, resource contention, eviction, or retries, stretching a long tail to the right. A mean barely moves in response to that tail. Percentiles look directly at where it sits.

This matters because the user who landed in the slow 1% had a genuinely bad experience regardless of how good the average looked. Production SLOs are usually written against tail latency for exactly this reason — which means a benchmark reporting only mean TTFT can look substantially better than what real users feel.

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

**QPS is completed requests per second.** Of all the throughput metrics, it maps most directly onto the question an operator actually asks: how many users can this deployment serve?

<!--
slug: cache-hit-rate
question: What is cache hit rate?
aliases: [hit rate, KV cache hit rate]
related: [cache-hit, prefix-caching, eviction-policy, chunk-size]
tier: 3
publish: yes
-->
### Cache hit rate

**Cache hit rate is the share of requested tokens that were served from cache rather than recomputed.** It is the mediating variable behind every other caching claim.

KV caching only helps when parts of a prompt repeat across requests — a shared system prompt, a repeated document, an earlier turn in the same conversation. Hit rate is the switch that determines whether caching does anything at all for a given request.

Higher hit rate compounds the benefit rather than merely extending it. A workload with heavy prompt overlap saves far more cumulative TTFT and throughput than a workload of all-unique one-shot prompts, where hit rate sits near zero and caching has essentially nothing to work with.

Nobody asks "what's my cache hit rate" the way they ask "how fast is my app" — but every latency and throughput claim routes through it. Hit rate is sensitive both to the workload's actual repetition pattern and to pool sizing, chunk size, and eviction policy, which means a benchmark that neither discloses nor controls it is reporting an unrepeatable number.

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

The two headline metrics trade off along this axis. Low concurrency means each request gets more GPU attention: higher interactivity, lower total throughput. High concurrency means the GPU juggles more requests: lower interactivity per user, higher total throughput.

The practical consequence for reading benchmarks: a result reporting only one of these, without stating the concurrency level it was measured at, is not comparable to another result measured at a different concurrency level — even on identical hardware.

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

That comparison is the bet a caching layer is fundamentally making — cheaper storage substituting for more expensive GPU memory and compute.

> **Sourcing note:** public numbers in this space are overwhelmingly vendor-sourced, including ours. Any figure we publish should be labeled as such, and independent GPU and storage pricing is worth gathering before we put dollar amounts in front of readers.

---

## Tier 4 — Engine and hardware knobs

<!--
slug: quantization
question: What is quantization in LLM inference?
aliases: [precision, dtype, fp8, kv cache dtype]
related: [model-weights, kv-cache, cache-hit-rate]
tier: 4
publish: yes
-->
### Quantization and precision

**Quantization is storing and computing numbers at lower precision to save memory and time.** Two separate settings get confused constantly:

- **Weight precision** (`--dtype`: fp16/bf16/fp8/fp4) — how the model's own parameters are stored and computed. Lower precision generally means faster matrix multiplication on hardware with dedicated support, speeding up both prefill and decode, independent of caching entirely.
- **KV-cache precision** (`--kv-cache-dtype`, a separate flag) — how the cached key/value tensors are stored, independent of weight precision. A lower-precision KV cache takes less memory per token, so more tokens fit in the same pool, directly raising the ceiling on hit rate — at the cost of a small dequantization step on every read.

Precision is also hardware-dependent in a way that surprises people. fp8 matrix multiplication is accelerated on GPUs with dedicated fp8 tensor cores; an older GPU without that support emulates it in software and can end up *slower* than just running bf16. The same flag behaves differently on different hardware.

And it is an approximation, not a free win. Below a certain point, depending on the model, lower precision measurably degrades output quality.

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

Because that combining step happens at *every single layer*, TP needs a very fast GPU-to-GPU link — NVLink rather than PCIe — to avoid being dominated by communication. This is why TP is typically confined to GPUs inside one physical machine.

It is not a free speedup. On a slow interconnect, adding more TP can make inter-token latency worse rather than better.

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

Communication happens only at stage handoffs, far less than TP requires, so PP tolerates slower interconnects and can span separate machines. The cost is pipeline "bubbles" — idle GPU time — unless enough requests are kept in flight to keep every stage busy.

<!--
slug: expert-parallelism
question: What is expert parallelism?
aliases: [EP, MoE parallelism]
related: [tensor-parallelism, pipeline-parallelism]
tier: 4
publish: yes
-->
### Expert parallelism (EP)

**Expert parallelism places different experts of a mixture-of-experts model on different GPUs**, routing each token to whichever GPU holds the expert its router selected.

The communication pattern is an all-to-all shuffle of tokens out to their assigned GPU and back — different in shape from TP's per-layer combine, and applicable only to MoE architectures.

Whichever parallelism type is used, splitting a model across N GPUs also splits the total memory pool across them, changing how much room is left per GPU for cached KV data.

<!--
slug: speculative-decoding
question: What is speculative decoding?
aliases: [draft model, speculative sampling]
related: [decode, inter-token-latency, decode-throughput]
tier: 4
publish: yes
-->
### Speculative decoding

**Speculative decoding uses a smaller "draft" model to propose several tokens ahead, which the main model then verifies in a single pass.**

It can meaningfully improve inter-token latency and decode throughput, and it is entirely independent of whether any KV cache is being reused. Worth naming because it belongs in any honest list of what moves these metrics — caching is not the only lever, and LMCache does not do this.

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

Two engines — or two versions of the same engine — running an identical model on identical hardware can differ in speed purely because of which attention kernel they use. Same math, different code, different speed.

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

**Interconnects are the links data travels over between GPUs, machines, and storage.** Which one is in play often matters more to a benchmark result than any software setting.

- **PCIe** — the standard bus connecting a GPU to the rest of its machine (CPU, RAM). Speed depends on generation and lane count.
- **NVLink** — a much faster GPU-to-GPU (and on some systems GPU-to-CPU) interconnect than PCIe, available only on hardware that supports it.
- **RDMA** — lets one machine read and write another machine's memory directly over the network, bypassing the remote CPU. Much faster and lower-latency than a conventional TCP/HTTP path. This is what LMCache's peer-to-peer adapter uses to pull cached data from another instance.

Any claim about cache retrieval being faster than recomputation is, underneath, a claim about which of these the data crossed.

---

## Tier 5 — Deployment and systems

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

**A process is one running instance of a program**, with its own private memory, sharing nothing with other processes unless explicitly wired together through a network call, shared memory, or a message queue.

Launch the same script in two terminal windows and you have two separate processes. A single vLLM plus LMCache deployment on one machine typically runs several at once — under `--tensor-parallel-size 4`, that is four separate GPU worker processes.

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

The reason to want several: a process often needs to do things that should not block each other. In LMCache, one thread accepts incoming lookup and store requests, another waits for GPU-to-CPU copy-completion signals, another periodically evicts old entries. On a single thread, one slow step would freeze all the others.

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

**A container is a lightweight, self-contained package of one process plus everything it needs to run** — code, dependencies, system libraries — built so it behaves identically regardless of which physical machine runs it.

Lighter than a virtual machine: a container shares the host's operating system kernel rather than emulating a whole separate OS, so many can run on one machine with little overhead. The standard analogy is a shipping container — contents stay consistent no matter which ship or truck carries it.

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

**A cluster is the whole collection of nodes managed together as one system.**

"Fleet" is the informal version of the same idea — the group of running replicas considered together, without implying the specific Kubernetes machinery underneath.

<!--
slug: replica
question: What is a replica in Kubernetes?
aliases: [replicas, Deployment, ReplicaSet, autoscaling]
related: [pod, cluster, load-balancer, cache-aware-routing]
tier: 5
publish: support
-->
### Replica (Deployment / ReplicaSet)

**A replica is one running copy of a service.** A Kubernetes Deployment or ReplicaSet is the object that declares "keep N copies of this pod running at all times" — the literal mechanism that creates and maintains them.

Autoscaling works by having something adjust that N up or down based on load.

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

**A Service is a stable network entry point that distributes incoming requests across whichever pods are currently running for a deployment.** It is the routing layer — the thing deciding which replica a given request actually reaches.

Plain Kubernetes load balancing has no concept of cache locality. It does not know or care which replica might already hold relevant cached data, which is precisely the gap [cache-aware routing](/glossary/cache-aware-routing) fills.

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

Where Kubernetes generically keeps N pods running, an Operator understands application-specific concerns: how to safely scale down a cache server without losing in-flight lookups, how to roll out a new engine version without dropping requests. It is an optional layer of smarter automation, not something every deployment has.

<!--
slug: liveness-probe
question: What are liveness and readiness probes?
aliases: [readiness probe, health check, healthz]
related: [pod, operator, replica]
tier: 5
publish: support
-->
### Liveness and readiness probes

**Probes are how Kubernetes periodically asks a running pod whether it is healthy**, usually by hitting a URL like `/healthz` and checking the response.

A failing **liveness** probe means Kubernetes assumes the pod is stuck and restarts it. A failing **readiness** probe means "do not route new traffic here yet" without necessarily restarting anything — for example while a pod is still loading model weights.

LMCache's `mp_coordinator` component exposes a liveness endpoint, which is concrete evidence that its fleet-coordination piece is built to be managed *by* Kubernetes rather than sitting adjacent to it.

<!--
slug: rest
question: What is REST?
aliases: [REST API, RESTful]
related: [process, load-balancer]
tier: 5
publish: support
-->
### REST

**REST is a common convention for how two programs talk over plain HTTP** — standardized verbs (`GET`, `POST`) against standardized URLs, so any two programs that both speak it can exchange data without custom communication code.

Think of it as a standardized survey form: as long as both sides agree on the format, neither has to invent a bespoke conversation.

---

## Open items before publication

1. **Technical review.** Every Tier 1–4 definition should be read by someone on the core team. The CacheBlend, parallelism, and quantization entries are the highest-risk for drift against the current release.
2. **Version-pin the specifics.** Default values (`chunk_size` 256, `max_local_cpu_size` 5 GB, LRU eviction), CacheBlend's model support list, and the ~15% recompute ratio should either be checked against the current release or rewritten to avoid stating a number that will age.
3. **Cost figures.** The TCO entry deliberately carries no dollar amounts. Do not add any before there is a non-vendor source.
4. **Decide the Tier 5 treatment.** These are support definitions, not audience-facing pages. Recommend index-only entries rather than standalone URLs.
5. **Auto-linking.** Once slugs are stable, the first mention of any term in a blog post should link to its page. That internal linking, not the glossary itself, is where most of the value comes from.
