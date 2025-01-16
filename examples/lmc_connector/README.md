# Offline inference on LLM
This will help with offline inference on vLLM + LMCache (experimental).  
The default context is a ffmpeg man page.  

Results are stores in offline_inference_outputs.jsonl. Each line is a call to generate function.  
Each line consists of two lists: 'user_inputs', 'generated_texts' and a float number 'time in seconds'.  
Each element in the two lists belong to one prompt in that call to generate function, and 'time in seconds' shows how much time this call takes.  
## Prerequisites
Your server should have at least 1 GPU.  

## Steps
1. ```LMCACHE_USE_EXPERIMENTAL=True LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 python3 offline_inference.py```  
## What to expect
LMCache should be able to reduce the generation time of the second generate call.  
