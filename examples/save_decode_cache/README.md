# Saving decode cache to boost multi-turn conversation
This will help with offline inference on vLLM + LMCache.  
The example consists of two runs.
The first run asks the LLM to write a lengthy document about FFmpeg.
The second run asks the LLM to score its own generation, which should be able to retrieve the decoded chunks.  
The third 
## Prerequisites
Your server should have at least 1 GPU.  

This will use port 65432(for LMCache).  
## Steps
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. ```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 python3 offline_inference.py```  
## What to expect
The second request should retrieve from LMCache, and have a better performance than the third request.  