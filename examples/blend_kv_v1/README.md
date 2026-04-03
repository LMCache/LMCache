# Examples vLLM + LMCache w. CacheBlend
LMCache should be able to reduce the generation time of the second and following calls (even though the reused KV cache is not a prefix).

## Compatibility Note
This example is tested and verified with vLLM v0.17.1 (V1 Engine Alpha).

## Required ad-hoc changes in vLLM (v0.17.1)
To enable CacheBlend functionality, certain internal vLLM structures must be registered with LMCache during the worker initialization process.

Please apply the following changes to `vllm/v1/worker/gpu_worker.py`:

### Register Model Instance
In the function `initialize_from_config(self, kv_cache_config: KVCacheConfig)`, add the model registration logic before the KV connector is initialized.
This is required because LMCBlenderBuilder needs to access the model runner during connector setup.

```python
def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    """Allocate GPU KV cache with the specified kv_cache_config."""

    # CacheBlend: register model with LMCache tracker before KV
    # connector init, because LMCBlenderBuilder.get_or_create() calls
    # VLLMModelTracker.get_model() during connector initialization.
    try:
        from lmcache.v1.compute.models.utils import VLLMModelTracker
        from lmcache.integration.vllm.utils import ENGINE_NAME
        VLLMModelTracker.register_model(
            ENGINE_NAME, self.model_runner.model)
    except ImportError:
        pass

    # Existing KV connector initialization follows...
```

## Running the Examples
### CPU offloading
- `python blend.py` - CacheBlend with CPU as backend
### Disk offloading
- `python blend.py --use-disk` - CachBlend with local disk as backend
