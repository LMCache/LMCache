`long_doc_permutator.py` is meant to test different edge cases and behaviors of the V2 implementation of the Blend Server.

At a high-level, this script sends many permutations of a set of contexts.

```text
# one permutation
[System Prompt] + [Doc 1] + [Doc 2] + ... + [Doc N]

# another permutation
[System Prompt] + [Doc 2] + [Doc N] + ... + [Doc 1]
```


So far there are five targeted behaviors / areas we want to stress test:

1. Blended Context Boundaries

In blend v2, without the delimiter (e.g. `# #`) between blended contexts, the boundaries between contexts can be tokenized differently.

Configurations aimed at the context boundaries are:
- `--num-contexts` (default: 5)

```text
# the boundaries may be tokenized differently
[Doc 1[ ]] + [[ ]Doc 2]
[Doc 2[ ]] + [[ ]Doc 1]
```

2. Eviction

The blend server keeps its own memory of contexts seen before. If the actual KV Cache in L1 or L2 is evicted, can the Blend Server properly "forget" them?

Configurations aimed at eviction are:
- `--num-permutations` (default: 10)

```text
# we enumerate permutations up to the specified count (capped at N!)
len(set(chosen permutations))
```

3. Chunk Homogeneity

In blend v2, the lookup and indexing of blended contexts is done through a rolling hash on the tokens in a sliding window the size of a chunk.
This means that if multiple windows are the same, one blended chunk could "overwrite" another.

Configurations aimed at chunk homogeneity are:
- `--vocab-size` (default: 8000)

```text
Vocab Pool = generate_vocab_pool(size=8000)

Each document is constructed by sampling from the vocab pool.
Smaller vocab = higher collision risk.
```

4. Prefix Domination

We also want to see how blend performs when most of the KV Cache reuse is still prefix based.
We aim to see both whether blend correctly triggers in the prefix case and whether there is any performance regression compared to regular prefix caching.

Configurations aimed at prefix domination are:
- `--system-prompt-length` (default: 1000)

```text
the same system prompt is used each time for each request. specify 0 for no system prompt
```

5. Concurrency

Can blend v2 handle concurrency?

Configurations aimed at concurrency are:
- `--max-inflight-requests` (default: 1)

```text
0 = flood all requests at once (throughput mode)
1 = sequential (TTFT mode)
N = up to N concurrent requests
```

## Examples

Each example isolates one stress axis while keeping the others at mild defaults.

### 1. Stress Context Boundaries

Many small contexts to maximize the number of inter-document boundaries the rolling hash must handle.

```bash
python long_doc_permutator.py \
    --num-contexts 20 \
    --num-permutations 10 \
    --context-length 500 \
    --system-prompt-length 1000 \
    --vocab-size 8000 \
    --max-inflight-requests 1 \
    --output-dir results/boundaries
```

### 2. Stress Eviction

Send many permutations to flood the blend server's memory and force eviction from L1/L2 cache.

```bash
python long_doc_permutator.py \
    --num-contexts 5 \
    --num-permutations 120 \
    --context-length 5000 \
    --system-prompt-length 1000 \
    --vocab-size 8000 \
    --max-inflight-requests 1 \
    --output-dir results/eviction
```

### 3. Stress Chunk Homogeneity

Use a tiny vocabulary so many sliding windows produce identical hashes, maximizing chunk collision risk.

```bash
python long_doc_permutator.py \
    --num-contexts 5 \
    --num-permutations 10 \
    --context-length 5000 \
    --system-prompt-length 1000 \
    --vocab-size 6 \
    --max-inflight-requests 1 \
    --output-dir results/homogeneity
```

### 4. Stress Prefix Domination

Make the system prompt very long relative to the contexts so the majority of KV reuse is prefix-based.

```bash
python long_doc_permutator.py \
    --num-contexts 5 \
    --num-permutations 10 \
    --context-length 2000 \
    --system-prompt-length 20000 \
    --vocab-size 8000 \
    --max-inflight-requests 1 \
    --output-dir results/prefix_domination
```

### 5. Stress Concurrency

Flood all requests at once to test blend v2 under parallel load.

```bash
python long_doc_permutator.py \
    --num-contexts 5 \
    --num-permutations 10 \
    --context-length 5000 \
    --system-prompt-length 1000 \
    --vocab-size 8000 \
    --max-inflight-requests 0 \
    --output-dir results/concurrency
```
