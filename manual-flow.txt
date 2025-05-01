# TL;DR

KV shape mismatch due to Deepseek V2 and V3 specific MLA (Multi-head Latent Attention)
KV optimizations that LMCache seems to be incompatible with right now.

# Set Up

Since the SGLang MMLU benchmark uses vllm.entrypoints.api_server instead of
vllm.entrypoints.openai.api_server, we first need to manually add a line
to `vllm/entrypoints/api_server.py`:

```python
app = await init_app(args, llm_engine)
assert engine is not None

app.state.engine_client = engine ## add this here!!

shutdown_task = await serve_http(
```

Make sure you have a simple LMCache configuration file set up: e.g. `lmc-cpu.yaml` with:

```
chunk_size: 256
local_cpu: True
max_local_cpu_size: 10
remote_serde: "naive"
```

After starting the vllm server (on localhost and port 8000), we can run
MMLU benchmarks (this will test the default 60 subjects) inside
`sglang/benchmark/mmlu` (make sure to download the data first):

```bash
python3 bench_other.py --backend vllm --host http://localhost     --port 8000 --parallel 16
```

# Diagnosing the Issue:

**Step 1:** First, we check if this error can be reproduced with a small dense model that does
not use Multi-head Latent Attention. The suspicion is that the
error is deepseek/MLA specific but we need to be sure.

Small Dense Model: llama 3.1 8B

1A. vllm v0 WITH LMCache
```bash
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=0 \
VLLM_USE_V1=0 \
LMCACHE_CONFIG_FILE=/home/samuelshen/lmc-cpu.yaml \
python3 -m vllm.entrypoints.api_server \
  --model /home/samuelshen/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --trust-remote-code \
  --served-model-name vllm_cpu_offload \
  --max-model-len 8192 \
  --max-seq-len-to-capture 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}'
```

See full results in `mmlu-results/v0_lmcache_llama.txt`

Summary: `Average accuracy: 0.683`

1B. vllm v0 WITHOUT LMCache
```bash
VLLM_USE_V1=0 \
python3 -m vllm.entrypoints.api_server \
  --model /home/samuelshen/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
  --trust-remote-code \
  --served-model-name vllm_cpu_offload \
  --max-model-len 8192 \
  --max-seq-len-to-capture 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --tensor-parallel-size 1
```

See full results in `mmlu-results/v0_llama.txt`

Summary: `Average accuracy: 0.684`


**Step 2:** We have confirmed that there is a trivial difference (0.683 vs 0.684)
between using LMCache and not using LMCache with a dense model like llama. In
`vllm/distributed/kv_transfer/kv_connector/utils.py`, we find a critical
section.

```python
# Deepseek's MLA (Multi-head Latent Attention) uses two different
# kv_cache shapes based on whether VLLM_MLA_DISABLE is set to 0.
# When VLLM_MLA_DISABLE=0 (default), forward absorb is applied,
# resulting in a kv_cache shape of [num_blks, blk_size, 1,
# kv_lora_rank + qk_rope_head_dim].
# When VLLM_MLA_DISABLE=1, standard FA is used instead, leading
# to a kv_cache shape of [2, num_blks, blk_size,
# num_key_value_heads / tp, qk_nope_head_dim + qk_rope_head_dim].
# For more details, see vllm/attention/backends/mla/common.py.
if self.is_deepseek_mla and self.use_mla_opt:
    head_size = model_config.kv_lora_rank + \
        model_config.qk_rope_head_dim
    num_heads = 1
elif self.is_deepseek_mla and not self.use_mla_opt:
    head_size = model_config.qk_nope_head_dim + \
        model_config.qk_rope_head_dim
else:
    head_size = getattr(model_config, "head_dim",
                        int(hidden_size // num_attention_heads))

return num_heads, head_size
```

We should try to reproduce on some V2 or V3 model. [V2 uses MLA](https://arxiv.org/pdf/2405.04434).

I will use `deepseek-ai/DeepSeek-V2-Lite`. As long as it uses MLA and we keep
`VLLM_MLA_DISABLE=0`, we should encounter the bug.

2A. vllm v0 WITHOUT LMCACHE

```bash
VLLM_USE_V1=0 \
VLLM_MLA_DISABLE=0 \
python3 -m vllm.entrypoints.api_server \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --served-model-name deepseek_test \
  --max-model-len 8192 \
  --max-seq-len-to-capture 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --tensor-parallel-size 2
```

See full results in `mmlu-results/v0_deepseek2.txt`

Summary: `Average accuracy: 0.578`

2B. vllm v0 WITH LMCACHE WITH `VLLM_MLA_DISABLE=1`

```
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=1 \
VLLM_USE_V1=0 \
LMCACHE_CONFIG_FILE=/home/samuelshen/lmc-cpu.yaml \
python3 -m vllm.entrypoints.api_server \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --served-model-name deepseek_test \
  --max-model-len 8192 \
  --max-seq-len-to-capture 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --tensor-parallel-size 2 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}'
```

See full results in `mmlu-results/v0_lmcache_deepseek2_no_mla.txt`

Summary: `Average accuracy: 0.577`

Indeed, disabling VLLM MLA yields the same result.

2C. vllm v0 WITH LMCACHE (PROBLEM RUN)

This was very buggy and would crash everytime I tried the benchmarks.

```bash
LMCACHE_USE_EXPERIMENTAL=True \
LMCACHE_TRACK_USAGE=false \
VLLM_MLA_DISABLE=0 \
VLLM_USE_V1=0 \
LMCACHE_CONFIG_FILE=/home/samuelshen/lmc-cpu.yaml \
python3 -m vllm.entrypoints.api_server \
  --model deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --served-model-name deepseek_test \
  --max-model-len 8192 \
  --max-seq-len-to-capture 2048 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --tensor-parallel-size 2 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both","kv_parallel_size":2}'
```

Server starts, but once benchmarks are run, error correctly appears:
`AssertionError: Only FlashAttention backend is supported for now.`

This is clearly intended behavior. But we will remove these assertions and see
what catastrophe enfolds.

As expected, the `vllm/vllm/attention/backends/mla/common.py` crashes on:
```
    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ):
```


Feel free to checkout the errors logs in `mmlu-results/v0_lmcache_deepseek2_errors.txt`