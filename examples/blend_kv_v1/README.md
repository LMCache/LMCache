# Examples vLLM + LMCache w. CacheBlend

> [!WARNING]
> This example uses the deprecated in-process CacheBlend integration. The vLLM
> patch below applies to vLLM v0.8.5 through v0.11.0 and does not apply to
> v0.12.0 or later. For current vLLM releases, use the
> [MP-mode CacheBlend guide](https://docs.lmcache.ai/kv_cache_optimizations/cacheblend.html).

LMCache should be able to reduce the generation time of the second and following calls (even though the reused KV cache is not a prefix).

## Some ad-hoc changes needed in vLLM
- In `vllm/vllm/v1/worker/gpu_worker.py`, comment out `ensure_kv_transfer_initialized(vllm_config)` in function `def init_worker_distributed_environment`.
- In the same file, add 
```
from lmcache.v1.compute.models.utils import VLLMModelTracker
from lmcache.integration.vllm.utils import ENGINE_NAME
        
VLLMModelTracker.register_model(ENGINE_NAME, self.model_runner.model)
ensure_kv_transfer_initialized(self.vllm_config)
```
at the end of the function `def load_model`.

## CPU offloading
- `python blend.py` - CacheBlend with CPU as backend
## Disk offloading
- `python blend.py --use-disk` - CacheBlend with local disk as backend
