# Note on hipify

1. Installation (after this, can install vLLM as usual)
```
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH # may or may not needed
PYTORCH_ROCM_ARCH="gfx942" TORCH_DONT_CHECK_COMPILER_ABI=1 CXX=hipcc BUILD_WITH_HIP=1 python3 -m pip install --no-build-isolation -e .
```

2. To run vLLM with LMcache
* Prepare a config yaml, lmcache_config.yaml
```
# 256 Tokens per KV Chunk
chunk_size: 256
# Redis host
remote_url: "redis://0.0.0.0:6379"
# Redis Sentinel hosts (for high availability)
# remote_url: "redis-sentinel://localhost:26379,localhost:26380,localhost:26381"
# LMCache Server host
# remote_url: "lm://localhost:65432"

# How to serialize and deserialize KV cache on remote transmission
remote_serde: "cachegen" # "naive" (default) or "cachegen"
```

* Start a redis server
```
docker run --rm --name lmcache-redis --network host redis
```

* Start vLLM server
```
VLLM_USE_V1=1 LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=lmcache_config.yaml HIP_VISIBLE_DEVICES=6,7 vllm serve meta-llama/Llama-3.3-70B-Instruct --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' --port 19999 --host 0.0.0.0 --gpu-memory-utilization 0.9 --quantization fp8 -tp 2
```

3. Tested models:
Qwen2-7B
Qwen3-8B (+ on-the-fly FP8 quantization)
Qwen3-30B-A3B-FP8
Llama3.3-70B (tp2, with on-the-fly FP8 quantization)
Qwen2.5-VL-7B-Instruct (text only, image doesn't work)
Mixtral 8x22B
