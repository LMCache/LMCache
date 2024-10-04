# Online chat with LLM
This will help with online chat on vLLM + LMCache.  
The default context is a ffmpeg man page.  
## Prerequisites
Your server should have at least 1 GPU.  

This will use the port 8000 (for vLLM) and port 65432(for LMCache).  
## Steps
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. In one terminal,  
```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 lmcache_vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000```  
Wait until it's ready.  
3.  ```python3 openai_chat_completion_client.py 8000```  
Then you can start to chat with the model.  
## What to expect
LMCache should be able to reduce the response delay since the second question.  
