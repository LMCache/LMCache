# Start an online chat with LLM
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. In one terminal,  
```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 lmcache_vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000```  
Wait until it's ready.  
3.  ```python3 openai_chat_completion_client.py 8000```  
Then you can start to chat with the model.  

