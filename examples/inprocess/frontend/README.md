# Online chat with frontend and context
This will help you set up vLLM + LMCache and a QA frontend.  
The default context is a ffmpeg man page.  
## Prerequisites
Your server should have at least 1 GPU.  

This uses port 8000 for vLLM, 8501 for the frontend, and 6379 for Redis.
## Steps
1. Start Redis and wait until it is ready:
   ```bash
   docker run --rm -p 6379:6379 redis:7
   ```
2. In one terminal,  
```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000 --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'```  
Wait until it's ready.  
3. Launch frontend.  
```pip install openai streamlit```  
```streamlit run frontend.py```  
Then open that URL of Streamlit app in browser.  
## What to expect
LMCache should be able to reduce the response delay since the second question.
