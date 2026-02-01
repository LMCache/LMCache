# DeepSeek-V3.2 Usage Guide

## Installing DeepGEMM

```bash
uv pip install git+https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3 --no-build-isolation
```

Note: DeepGEMM is used in two places: MoE and MQA logits computation. It is necessary for MQA logits computation. If you want to disable the MoE part, you can set `VLLM_USE_DEEP_GEMM=0` in the environment variable. Some users reported that the performance is better with `VLLM_USE_DEEP_GEMM=0`, e.g. on H20 GPUs. It might be also beneficial to disable DeepGEMM if you want to skip the long warmup.

## Installing vLLM and LMCache 

```bash
uv venv
source .venv/bin/activate
uv pip install vllm --extra-index-url https://wheels.vllm.ai/nightly
uv pip install lmcache
```

## Launching DeepSeek-V3.2 on 8xH200


- The chat-template changes in the DeepSeek-V3.2 are quite significant. vLLM adapts to this through `--tokenizer-mode deepseek_v32`.


```bash
  vllm serve deepseek-ai/DeepSeek-V3.2 \
   --tensor-parallel-size 8 \
   --tokenizer-mode deepseek_v32 \
   --tool-call-parser deepseek_v32 \
   --enable-auto-tool-choice \
   --reasoning-parser deepseek_v3
   --no-enable-prefix-caching \
   --port 8000 --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
```

Note: Fail to serve now 

