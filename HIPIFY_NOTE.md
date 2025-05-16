# Note on hipify
1. Run the hipify command
```
python3 hipify.py -p csrc/ -o csrc_hip/ csrc/*
```

2. Edit 2 files.
* update mem_kernels.hip
```
// #include <cuda_fp8.h>
#include <hip/hip_fp8.h>
```

* update pos_kernels.hip (maybe there's easier way)
```
// replace the original apply_token_rotary_embedding_fused
template <typename scalar_t, bool IS_NEOX>
inline __device__ void apply_token_rotary_embedding_fused(
    scalar_t* __restrict__ arr,
    const scalar_t* __restrict__ old_cos_ptr,
    const scalar_t* __restrict__ old_sin_ptr,
    const scalar_t* __restrict__ new_cos_ptr,
    const scalar_t* __restrict__ new_sin_ptr,
    int rot_offset, int embed_dim) {
  int x_index, y_index;
  scalar_t old_cos, old_sin;
  scalar_t new_cos, new_sin;

  if (IS_NEOX) {
    // GPT-NeoX style rotary embedding.
    x_index = rot_offset;
    y_index = embed_dim + rot_offset;

    // Replace __ldg with direct array access
    old_cos = old_cos_ptr[x_index];
    old_sin = old_sin_ptr[x_index];
    
    new_cos = new_cos_ptr[x_index];
    new_sin = new_sin_ptr[x_index];
  } else {
    // GPT-J style rotary embedding.
    x_index = 2 * rot_offset;
    y_index = 2 * rot_offset + 1;

    // Replace __ldg with direct array access
    old_cos = old_cos_ptr[x_index / 2];
    old_sin = old_sin_ptr[x_index / 2];

    new_cos = new_cos_ptr[x_index / 2];
    new_sin = new_sin_ptr[x_index / 2];
  }

  const scalar_t x = arr[x_index];
  const scalar_t y = arr[y_index];

  // This part applies the "reverse" rotation with old_cos/old_sin,
  // then the "forward" rotation with new_cos/new_sin.
  // The variable names x_reverse and y_reverse might be a bit confusing
  // if one expects them to be the original values, but they are intermediate values
  // after applying the first part of a two-step rotation.
  // Let's trace the logic:
  // 1. Rotate (x, y) by -theta_old to get (x_reverse, y_reverse)
  //    x_orig = x_current * cos_old + y_current * sin_old
  //    y_orig = y_current * cos_old - x_current * sin_old
  //    This seems to be the structure of your x_reverse, y_reverse calculation.
  const scalar_t x_reverse = x * old_cos + y * old_sin;
  const scalar_t y_reverse = y * old_cos - x * old_sin;

  // 2. Rotate (x_reverse, y_reverse) by +theta_new to get the final values
  //    x_final = x_orig * cos_new - y_orig * sin_new
  //    y_final = y_orig * cos_new + x_orig * sin_new
  //    This matches the structure of your final assignment.
  arr[x_index] = x_reverse * new_cos - y_reverse * new_sin;
  arr[y_index] = y_reverse * new_cos + x_reverse * new_sin;
}
```

3. Edit setup.py (as in this branch example)

4. Installation (after this, can install vLLM as usual)
```
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH # may or may not needed
PYTORCH_ROCM_ARCH="gfx942" TORCH_DONT_CHECK_COMPILER_ABI=1 CXX=hipcc BUILD_WITH_HIP=1 python3 -m pip install --no-build-isolation -e .
```

5. To run vLLM with LMcache
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
LLM_USE_V1=1 LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=lmcache_config.yaml HIP_VISIBLE_DEVICES=6,7 vllm serve meta-llama/Llama-3.3-70B-Instruct --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' --port 19999 --host 0.0.0.0 --gpu-memory-utilization 0.9 --quantization fp8 -tp 2
```

6. Tested models:
Qwen2-7B
Qwen3-8B (+ on-the-fly FP8 quantization)
Qwen3-30B-A3B-FP8
Llama3.3-70B (tp2, with on-the-fly FP8 quantization)
Qwen2.5-VL-7B-Instruct (text only, image doesn't work)
Mixtral 8x22B
