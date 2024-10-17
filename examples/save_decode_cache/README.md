# Saving decode cache to boost multi-turn conversation
This will help with offline inference on vLLM + LMCache.  
The example consists of two runs.  
The first run asks the LLM to write a lengthy document about FFmpeg.  
The second run asks the LLM to score its own generation, which should be able to retrieve the decoded chunks.  
## Prerequisites
Your server should have at least 1 GPU.  

## Steps
1. ```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 python3 save_decoding.py```  
## What to expect
The loggings of the second request should show "INFO LMCache: Retrieved x chunks" with x > 0.  
(Without saving decoding cache, the short prefill prompt should cause x == 0).  