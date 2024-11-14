# KV blending example
Offline and online examples for KV blending.  
The KV blending functionality is enabled by setting `enable_blending: True` in the configuration yaml.  
## A note for tensor parallelism
With TP > 1, run with  
```
LMCACHE_CONFIG_FILE=xxx.yaml VLLM_WORKER_MULTIPROC_METHOD=spawn python3 yy.py
```
(Add VLLM_WORKER_MULTIPROC_METHOD=spawn)  

