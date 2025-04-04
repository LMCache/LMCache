<div align="center">
<img src="https://github.com/user-attachments/assets/a0809748-3cb1-4732-9c5a-acfa90cc72d1" width="720" alt="lmcache logo">
</a>
</div>

| [**Blog**](https://lmcache.github.io) | [**Documentation**](https://docs.lmcache.ai/) | [**Join Slack**](https://join.slack.com/t/lmcacheworkspace/shared_invite/zt-2viziwhue-5Amprc9k5hcIdXT7XevTaQ) | [**Interest Form**](https://forms.gle/mQfQDUXbKfp2St1z7) | [**Official Email**](contact@lmcache.ai) |

# 💡 What is LMCache?

TL;DR - Redis for LLMs. 

LMCache is a **LLM** serving engine extension to **reduce TTFT** and **increase throughput**, especially under long-context scenarios. By storing the KV caches of reusable texts across various locations including (GPU, CPU DRAM, Local Disk), LMCache reuse the KV caches of **_any_** reused text (not necessarily prefix) in **_any_** serving engine instance. Thus, LMCache saves precious GPU cycles and reduces response delay for users.  

By combining LMCache with vLLM, LMCaches achieves 3-10x delay savings and GPU cycle reduction in many LLM use cases, including multi-round QA and RAG.

Try LMCache with pre-built vllm docker images [here](https://docs.lmcache.ai/getting_started/docker.html).

# 🚀 Performance snapshot
![image](https://github.com/user-attachments/assets/7db9510f-0104-4fb3-9976-8ad5d7fafe26)

# 💻 Installation and Quickstart

Please refer to our detailed documentation for [LMCache V1](https://docs.lmcache.ai/getting_started/installation.html#install-from-source-v1) and [LMCache V0](https://docs.lmcache.ai/getting_started/installation.html#install-from-source-v0)

# Interested in Connecting?
Fill out the interest form and our team will reach out to you!
https://forms.gle/mQfQDUXbKfp2St1z7

# 🛣️ News and Milestones

- [x] LMCache V1 with vLLM integration with following features is live 🔥
  * High performance CPU KVCache offloading
  * Disaggregated prefill
  * P2P KVCache sharing
- [x] LMCache is supported in the [vLLM production stack ecosystem](https://github.com/vllm-project/production-stack/tree/main) 
- [x] User and developer documentation
- [x] Stable support for non-prefix KV caches
- [x] Support installation through pip install and integrate with latest vLLM
- [x] First release of LMCache 


# 📖 Blogs and documentations

Our latest [blog posts](https://lmcache.github.io) and the [documentation](https://docs.lmcache.ai/) pages are available online

# Community meeting

The community meeting for LMCache is co-hosted with the community meeting for the [vLLM production stack project](https://github.com/vllm-project/production-stack/tree/main). 

Meeting Details:

- Tuesdays at 4:00 PM PT – [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1j1gO2PcFQLBi98fq4djEEiqOK_oHgq9j&export=download)

- Tuesdays at 8:00 AM PT – [Add to Calendar](https://drive.usercontent.google.com/u/0/uc?id=1xdkxpg-OpxkuLqjegHQhihwBM9koFvSh&export=download)

Meetings alternate weekly between the two times. All are welcome to join!

## Contributing

We welcome and value any contributions and collaborations.  Please check out [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.


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

## License

This project is licensed under Apache License 2.0. See the [LICENSE](LICENSE) file for details.

