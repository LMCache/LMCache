<div align="center">
<img src="https://github.com/user-attachments/assets/a0809748-3cb1-4732-9c5a-acfa90cc72d1" width="720" alt="lmcache logo">
</a>
</div>


# 💡 What is LMCache?
LMCache lets LLMs prefill each text only once. By storing the KV caches of all reusable texts, LMCache can reuse the KV caches of **_any_** reused text (not necessarily prefix) in **_any_** serving engine instance. It thus reduces prefill delay, i.e., time to first token (TTFT), as well as saves the precious GPU cycles. 

By combining LMCache with vLLM, LMCaches achieves 3-10x delay savings and GPU cycle reduction in many LLM use cases, including multi-round QA and RAG.

Try LMCache with pre-built vllm docker images [here](https://github.com/LMCache/demo).

# 🚀 Performance snapshot
![image](https://github.com/user-attachments/assets/7db9510f-0104-4fb3-9976-8ad5d7fafe26)



# 💻 Quickstart

LMCache provides the integration to the latest vLLM (0.6.1.post2). To install LMCache, use the following command:
```bash
# requires python >= 3.10 and nvcc >= 12.1
pip install lmcache lmcache_vllm
```

LMCache has the same interface as vLLM (both online serving and offline inference).
To use the online serving, you can start an OpenAI API-compatible vLLM server with LMCache via:
```bash
lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8
```

To use vLLM's offline inference with LMCache, just simply add `lmcache_vllm` before the import to vLLM components. For example
```python
import lmcache_vllm.vllm as vllm
from lmcache_vllm.vllm import LLM 
```

More detailed documentation will be available soon.

## - Sharing KV cache across multiple vLLM instances

LMCache supports sharing KV across different vLLM instances by the `lmcache.server` module. Here is a quick guide:

```bash
# Start lmcache server
lmcache.server localhost 65432
```

Then, start two vLLM instances with the LMCache config file
```bash
wget https://raw.githubusercontent.com/LMCache/LMCache/refs/heads/dev/examples/example.yaml

# start the first vLLM instance
LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8 --port 8000

# start the second vLLM instance
LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=1 lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8 --port 8001
```


## - What's next
We also provide multiple docker-based demos at [🔗LMCache-demos repo](https://github.com/LMCache/demo). The demos cover the following use cases:
- Share KV caches across multiple serving engines [(🔗link)](https://github.com/LMCache/demo/tree/master/demo2-multi-node-sharing)
- Loading non-prefix KV caches for RAG [(🔗link)](https://github.com/LMCache/demo/tree/master/demo3-KV-blending)

# 🛣️ Incoming Milestones

- [x] First release of LMCache 
- [x] Support installation through pip install and integrate with latest vLLM
- [ ] Stable support for non-prefix KV caches
- [ ] User and developer documentation

# 📖 Blogs and papers
LMCache is built on two key techniques:
1. [**CacheGen [SIGCOMM'24]**](https://arxiv.org/abs/2310.07240): A KV-cache compression system that encodes KV caches into compact bitstreams.
2. [**CacheBlend [EuroSys'25]**](https://arxiv.org/abs/2405.16444): A KV-cache blending system that dynamically composes new KV caches from smaller ones.

Please read our [blog posts](https://lmcache.github.io) for more details.


