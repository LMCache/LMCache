# Start an online chat with LLM with frontend
1.  ```lmcache_server localhost 65432```  
And wait until it's ready.  
2. In one terminal,  
```LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 lmcache_vllm serve mistralai/Mistral-7B-Instruct-v0.2 --gpu-memory-utilization 0.8 --port 8000```  
Wait until it's ready.  
3. Launch frontend.  
```pip install openai streamlit```  
```streamlit run frontend.py```  
Then open that URL of Streamlit app in browser.  
