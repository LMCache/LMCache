# Token Dropping Examples using LMCache SDK

#### We provide ready-to-run Google Collab examples for token dropping!!! [[Link]](https://drive.google.com/drive/folders/1ILctdh_Lf51qDUf1v00osRfoylfBIlOD?usp=share_link)

> Caveat: each notebook may take ~15 minutes to run.

> Join LMCache Slack and contact us at #sig-kv-research if you also want to ship your jupyter notebook to LMCache, or to our google collab examples.

## Running example on your own GPU

Long prompts create large KV caches that eat up GPU memory and limit how many
requests fit in a batch. Smaller batch means lower decode throughput. To 
improve decode throughput, we then need to stuff more requests in a batch.
*Token dropping*, analogous to its name, select tokens to drop (by half in
these examples) to shrink each request's KV cache and improve decode
throughput by 1.5-1.7x. The example also demonstrates that the generation
accuracy is unaffected, even improved, by a good token dropping algorithm 
(SnapKV was chosen for this demonstration).

These examples use the LMCache SDK to do this: the SDK **retrieves** a 
request's cached tensors, **modifies** them, and **stores** them back for vLLM
to decode from. Users only need to supply the token dropping function, and the
SDK's batch and stream APIs does the job in an offline manner.

There is also an example meant to be run in Google Colab's GPU T4 which uses a
smaller model and a shorter dataset to fit the T4's memory.

## Examples

| Notebook | Strategy | Needs query tensor? |
| --- | --- | --- |
| [random_token_dropping.ipynb](./random_token_dropping.ipynb) | Drops a random subset of past tokens. Uses the KV cache only. | No |
| [snapkv_token_dropping.ipynb](./snapkv_token_dropping.ipynb) | SnapKV: keeps the first and last window, as well as the tokens the recent-window queries attend to most. Needs each request's query tensor to score importance. | Yes |
| [rkv_token_dropping.ipynb](./rkv_token_dropping.ipynb) | R-KV: SnapKV's importance term minus a **redundancy** term, so a token is kept only if it is both attended-to and says something new. Three optimized variants live in [rkv_variants/](./rkv_variants). | Yes |

R-KV's redundancy term compares all pairs of keys, which is $O(n^2)$ in both
time and memory. [rkv_token_dropping.ipynb](./rkv_token_dropping.ipynb)
implements it the way the paper describes it; the three notebooks in
[rkv_variants/](./rkv_variants) are standalone alternatives that make it
cheaper:

| Notebook | Redundancy term | Exact? | VRAM |
| --- | --- | --- | --- |
| [rkv_variant_cpu_exact.ipynb](./rkv_variants/rkv_variant_cpu_exact.ipynb) | Rewritten as one inner product, so the $n \times n$ matrix never exists. | Yes | None |
| [rkv_variant_gpu_exact.ipynb](./rkv_variants/rkv_variant_gpu_exact.ipynb) | The same rewrite, tiled over rows on the GPU. Fastest of the four. | Yes | ~1.55 GiB |
| [rkv_variant_buffered_cpu.ipynb](./rkv_variants/rkv_variant_buffered_cpu.ipynb) | Keys compared only inside a chunk, making the matrix block-diagonal. | No, ~96% token agreement | None |

[snapkv_colab.ipynb](./snapkv_colab.ipynb) is a demonstration done in Google
Colab Notebook, which can also be accessed here:
[Google Colab SDK Examples](https://drive.google.com/drive/folders/1ILctdh_Lf51qDUf1v00osRfoylfBIlOD?usp=share_link).

## Prerequisites

* **GPU.** To see token dropping raise the decode batch size, tune
  `--gpu-memory-utilization` together with the number of requests. These
  examples were run on a single RTX 6000 PRO.
* **LMCache**. The SDK can use either shared memory transport or pickle 
  transport. This example uses shared memory. To transfer the query tensors,
  pass `--enable transfer_query` when starting LMCache.
* **vLLM** with below patch.

### Installing LMCache

This example uses recent `dev`-branch features (`--enable transfer_query`,
engine-driven transfer), so install LMCache **from source** rather than a
released wheel:

```sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch                        # install torch FIRST
uv pip install -e . --no-build-isolation    # then build LMCache
```

A few things that commonly trip people up:

* **torch must be installed before LMCache, and you must pass
  `--no-build-isolation`.** The native CUDA extensions in `csrc/` compile
  against the installed torch's headers; without `--no-build-isolation` pip
  builds in an isolated environment that cannot see torch, and the build fails.
* **`nvcc` (CUDA toolkit) must match your torch CUDA version**, otherwise the
  `.cu` files fail to compile. Dependencies for CUDA 12 vs 13 are auto-selected
  from `requirements/`.
* **Rebuild after pulling `dev`.** `-e` (editable) makes Python changes take
  effect immediately, but native-extension changes do **not** — if `csrc/` or
  the native enums/kernels changed upstream, re-run
  `uv pip install -e . --no-build-isolation`, otherwise you can hit errors like
  `AttributeError: ... 'EngineKVFormat' has no attribute ...` from a stale `.so`.
* **Use `uv pip` (or the venv's `pip`), not the system `pip`**, which may be
  marked externally-managed (PEP 668) and refuse to install.

Build variants: `NO_NATIVE_EXT=1` (pure Python, no extensions),
`NO_GPU_EXT=1 ... --no-build-isolation` (CPU C++ only), `BUILD_WITH_HIP=1`
(AMD ROCm/HIP).

### vLLM patch to expose intermediate tensor

Many token dropping algorithms need the query tensor to rank the importance of
tokens. SnapKV is one of it. vLLM does not expose the intermediate tensors to 
the KV connector by default. A 10-line change adds it.

First install a vLLM version this patch has been tested against (0.23.0 through
0.25.1). The patch touches **only two Python files** — no recompilation is
needed; just restart vLLM after applying it.

If you have a vLLM **source checkout** (git tree), apply with `git apply`:

```sh
cd /path/to/vllm
git apply /path/to/LMCache/examples/token_dropping/vllm-export-intermediate-tensors.diff
```

If you installed vLLM from a **wheel** (`pip install vllm`), there is no git
tree to patch. Apply the diff directly to the installed package instead:

```sh
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
patch -p1 -d "$VLLM_DIR/.." < /path/to/LMCache/examples/token_dropping/vllm-export-intermediate-tensors.diff
```

When booting up vLLM, activate the code path by adding 
`lmcache.mp.transfer_intermediate_tensors` to the connector config.
By default, the QRingBuffer, a temporary staging buffer for containing
query tensor, has the capacity to hold the query tensor of 2 forward
passes. However, it can also be configured via `"lmcache.mp.q.ring_depth":2`.

```json
--kv-transfer-config '{
    "kv_connector": "LMCacheMPConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "lmcache.mp.port": 6555,
        "lmcache.mp.transfer_intermediate_tensors": true,
        "lmcache.mp.q.ring_depth":2
    }
}'
```

The random-dropping example, being the simplest example that demonstrates 
decode throughput improvement, does **not** need this patch or flag. It only
works with the KV cache.

## Dataset

The notebooks load the 
[`raniayu/token-dropping-demo`](https://huggingface.co/datasets/raniayu/token-dropping-demo)
dataset. The samples included in this dataset are taken from
[LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) by
choosing 30 examples whose prompt is closest to 10240 tokens (Qwen3-8B tokenizer).

The Google Colab notebook loads a shorter dataset, adjusted to GPU T4's capacity, 
[`raniayu/token-dropping-demo-short`](https://huggingface.co/datasets/raniayu/token-dropping-demo-short).
This short dataset is taken from 
[ehovy/race](https://huggingface.co/datasets/ehovy/race) 
by choosing 10 examples whose prompt length is closest to 1024 tokens and
having unique prefix contexts.
