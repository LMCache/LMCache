# Offline inference on LLM
This will help with offline inference on vLLM + LMCache.  
No context is given to LLM here.  
## Prerequisites
Your server should have at least 1 GPU.  

This will use port 65432(for LMCache).  
## Steps
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. ```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 python3 offline_inference.py```  
You can also modify the model and prompts inside this file.  
## What to expect
It should print outputs of the given prompts.  
