# Offline inference on LLM
This will help with offline inference on vLLM + LMCache.  
The default context is a ffmpeg man page.  
## Prerequisites
Your server should have at least 1 GPU.  

This will use port 65432(for LMCache).  
## Steps
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. ```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 python3 offline_inference.py```  
## What to expect
LMCache should be able to reduce the generation time of the second generate call.  
