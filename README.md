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
We provide a docker-based quickstart demo in the folder [`examples/`](https://github.com/LMCache/LMCache/tree/dev/examples). This quickstart lets you start a serving engine (vLLM) with LMCache and then query the serving engine with a long context.

## - Prerequisites

First, clone and cd into the LMCache repo with 
```bash
git clone https://github.com/LMCache/LMCache && cd LMCache
```

To run the quickstart demo, your server should have 1 GPU and the [docker environment](https://docs.docker.com/engine/install/) with the [nvidia-runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed. 

You may need sudo access to run the docker depending on the server configuration.

This demo will use the port 8000 (for vLLM) and 8501 (for the frontend).

## - Start the serving engine with LMCache

Start the docker-based serving engine by:
```bash
bash examples/quickstart.sh
```

The vLLM serving engine is ready after you see the following lines in the log:
<img width="630" alt="image" src="https://github.com/user-attachments/assets/b0f3cef5-4926-4d5b-9fe2-99d6981decd2">

## - Start the frontend

The quickstart comes with a frontend. To run the frontend, use:

```bash
pip install openai streamlit
streamlit run examples/quickstart-frontend.py
```

You should be able to access the frontend from your browser at `http://<your server's IP>:8501`

The first query has a long TTFT because the server needs to prefill the long context. But once the first quey finishes, the TTFT of all future queries will be much lower as LMCache shares the KV cache to vLLM which can then skip the prefill of the long context.

## - What's next
We provide multiple demos at [🔗LMCache-demos repo](https://github.com/LMCache/demo). The demos cover the following use cases:
- Share KV caches across multiple serving engines [(🔗link)](https://github.com/LMCache/demo/tree/master/demo2-multi-node-sharing)
- Loading non-prefix KV caches for RAG [(🔗link)](https://github.com/LMCache/demo/tree/master/demo3-KV-blending)

# 🛣️ Project Milestones

- [x] First release of LMCache 
- [ ] Support installation through pip install
- [ ] Integration with latest vLLM

# 📖 Blogs and papers
LMCache is built on two key techniques:
1. [**CacheGen [SIGCOMM'24]**](https://arxiv.org/abs/2310.07240): A KV-cache compression system that encodes KV caches into compact bitstreams.
2. [**CacheBlend [EuroSys'25]**](https://arxiv.org/abs/2405.16444): A KV-cache blending system that dynamically composes new KV caches from smaller ones.

Please read our [blog posts](https://lmcache.github.io) for more details.


