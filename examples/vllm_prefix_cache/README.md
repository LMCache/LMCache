# LMCache with vllm automatic prefix caching
This shows how LMCache works together with vllm automatic prefix caching.  
The first run asks serving engine 1 with a shorter context, which is truncated from the longer context. This will put a shorter KV cache into vllm automatic prefix caching of serving engine 1.  
The second run asks serving engine 2 with a longer context. This will put a longer KV cache into LMCache.  
The third run asks serving engine 1 with a longer context. This should make use of the vllm internal cache for the shorter prefix cache, and also retrieve from LMCache for the longer prefix cache.  
## Prerequisites
Your server should have at least 2 GPUs.  

This will use port 65432(for LMCache).  
## Steps
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2.  ```python3 vllm_prefix_cache.py```  
## What to expect
The loggings of LMCache "Injected token number:" should show that, fewer tokens are injected (from 25K to < 1k) into vllm KV cache on the third run, because of vllm internal cache.  

