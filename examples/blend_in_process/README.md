# CacheBlend with vLLM + LMCache (in-process mode)
LMCache should be able to reduce the generation time of the second and following calls (even though the reused KV cache is not a prefix).

This example runs LMCache in in-process mode, where the cache engine lives inside the vLLM
worker process. See the [blending documentation](https://docs.lmcache.ai/kv_cache_optimizations/blending.html)
for the configuration reference.

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
- `python blend.py --use-disk` - CachBlend with local disk as backend