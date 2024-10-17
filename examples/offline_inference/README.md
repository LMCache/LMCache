# Offline inference on LLM
This will help with offline inference on vLLM + LMCache.  
The default context is a ffmpeg man page.  

Results are stores in offline_inference_outputs.jsonl. Each line is a call to generate function.  
Each line consists of two lists: 'user_inputs', 'generated_texts' and a float number 'time in seconds'.  
Each element in the two lists belong to one prompt in that call to generate function, and 'time in seconds' shows how much time this call takes.  
## Prerequisites
Your server should have at least 1 GPU.  

This will use port 65432(for LMCache).  
## Steps
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. ```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 python3 offline_inference.py```  
## What to expect
LMCache should be able to reduce the generation time of the second generate call.  
The total time of processing the request(end to end request time) can improve by more than 3x, for example, 7s to 2s.  
(The exact number depends on hardware).  