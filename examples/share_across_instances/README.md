# Sharing KV cache across multiple vLLM instances
This shows how to share KV across different vLLM instances using LMCache.  
## Prerequisites
Your server should have at least 2 GPUs.  

This will use the port 8000 and 8001(for vLLM) and port 65432(for LMCache).  
## Steps
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. In one terminal,  
```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000```  
In ANOTHER terminal,   
```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=1 python3 -m lmcache_vllm.vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8001```  
And wait until both of them are ready.  
3. Warm up the engine.  
```python3 warmup_online.py 8000 8001```  
warmup_online.py is in the root of examples directory.  
3.  ```python3 ask_ffmpeg_question.py 8000```  
Wait until generation completes, then  
```python3 ask_ffmpeg_question.py 8001```  
The TTFT and Total time of the second question are expected to reduce.  
## What to expect
The TTFT and Total time of the second question should reduce.  
For example, TTFT can drop from 7.4s to 4.2s, depending on hardware and network.  

