<div align="center">
<img src="https://github.com/user-attachments/assets/a0809748-3cb1-4732-9c5a-acfa90cc72d1" width="720" alt="lmcache logo">
</a>
</div>

| [**Blog**](https://lmcache.github.io) | [**Documentation**](https://docs.lmcache.ai/) | [**Join Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ) | [**Interest Form**](https://forms.gle/mQfQDUXbKfp2St1z7) | [**Official Email**](contact@lmcache.ai) |

# 💡 What is LMCache?
LMCache is a **LLM** serving engine extension to **reduce TTFT** and **increase throughput**, especially under long-context scenarios. By storing the KV caches of reusable texts across various locations including (GPU, CPU DRAM, Local Disk), LMCache reuse the KV caches of **_any_** reused text (not necessarily prefix) in **_any_** serving engine instance. Thus, LMCache saves precious GPU cycles and reduces response delay for users.  

By combining LMCache with vLLM, LMCaches achieves 3-10x delay savings and GPU cycle reduction in many LLM use cases, including multi-round QA and RAG.

Try LMCache with pre-built vllm docker images [here](https://github.com/LMCache/demo).

# 🚀 Performance snapshot
![image](https://github.com/user-attachments/assets/7db9510f-0104-4fb3-9976-8ad5d7fafe26)

# 💻 Quickstart

LMCache provides the integration to the latest vLLM (0.6.2). To install LMCache, use the following command:
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
lmcache_server localhost 65432
```

Then, start two vLLM instances with the LMCache config file
```bash
wget https://raw.githubusercontent.com/LMCache/LMCache/refs/heads/dev/examples/example.yaml

# start the first vLLM instance
LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=0 lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8 --port 8000

# start the second vLLM instance
LMCACHE_CONFIG_FILE=example.yaml CUDA_VISIBLE_DEVICES=1 lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8 --port 8001
```


# - What's next
We also provide multiple docker-based demos at [🔗LMCache-demos repo](https://github.com/LMCache/demo). The demos cover the following use cases:
- Share KV caches across multiple serving engines [(🔗link)](https://github.com/LMCache/demo/tree/master/demo2-multi-node-sharing)
- Loading non-prefix KV caches for RAG [(🔗link)](https://github.com/LMCache/demo/tree/master/demo3-KV-blending)

# Interested in Connecting?
Fill out the interest form and our team will reach out to you!
https://forms.gle/mQfQDUXbKfp2St1z7

# 🛣️ Incoming Milestones

- [x] First release of LMCache 
- [x] Support installation through pip install and integrate with latest vLLM
- [ ] Stable support for non-prefix KV caches
- [ ] User and developer documentation

# 📖 Blogs and documentations

Our [blog posts](https://lmcache.github.io) and [documentations](https://docs.lmcache.ai/) are available online

# Community meeting

- :link: Meeting link - https://uchicago.zoom.us/j/91454186439?pwd=Qu3IMJH7c83Qbg9hHsXZ3BxzLaEFoF.1
- :page_facing_up: Community Meeting Document - https://docs.google.com/document/d/1SnCKnB2UFBUyPhIpL9zzdZsn_hGp50spoZue-2SoxJY/edit?usp=sharing
- 🗓️ Calendar - https://calendar.app.google/rsu7Xgq4y4y5YuDj7


## Citation
If you use LMCache for your research, please cite our papers:

```
@inproceedings{liu2024cachegen,
  title={Cachegen: Kv cache compression and streaming for fast large language model serving},
  author={Liu, Yuhan and Li, Hanchen and Cheng, Yihua and Ray, Siddhant and Huang, Yuyang and Zhang, Qizheng and Du, Kuntai and Yao, Jiayi and Lu, Shan and Ananthanarayanan, Ganesh and others},
  booktitle={Proceedings of the ACM SIGCOMM 2024 Conference},
  pages={38--56},
  year={2024}
}

@article{cheng2024large,
  title={Do Large Language Models Need a Content Delivery Network?},
  author={Cheng, Yihua and Du, Kuntai and Yao, Jiayi and Jiang, Junchen},
  journal={arXiv preprint arXiv:2409.13761},
  year={2024}
}

@article{yao2024cacheblend,
  title={CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion},
  author={Yao, Jiayi and Li, Hanchen and Liu, Yuhan and Ray, Siddhant and Cheng, Yihua and Zhang, Qizheng and Du, Kuntai and Lu, Shan and Jiang, Junchen},
  journal={arXiv preprint arXiv:2405.16444},
  year={2024}
}
```

