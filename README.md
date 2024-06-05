# LMCache core library

## Installation

```bash
pip install -e .
```


## Testing instructions

We have integrated LMCache into vLLM. The following instructions will help test
its functionality locally or in a containerized environment.

### Envisioned architecture

<img width="817" alt="image" src="https://github.com/LMCache/LMCache/assets/25103655/ab64f84d-26e1-46ce-a503-e7e917b618bc">

### New args to vLLM

To use LMCache in vLLM with different configurations, we added the following commandline
arguments to vLLM runtime:

```
  --enable-lmcache      Enable LMCache engine
  --lmcache-local-cache LMCACHE_LOCAL_CACHE_TYPE
                        Set the local cache backeend of lmcache, can be 'cuda' or 'cpu'
  --lmcache-remote-cache LMCACHE_REMOTE_URL
                        Set the url the remote cache backend of lmcache, currently support redis (redis://<host>:<port>), or lmcache-
                        server (lm://<host>:<port>)
  --lmcache-chunksize LMCACHE_CHUNKSIZE
                        Set the chunksize of lmcache engine
```

### Local testing

To vLLM + LMCache locally, please follow the following steps:

**NOTE:** We highly recommend running the installation in a newly created python virtual environment

Install LMCache core library
```bash
# clone and install LMCache
git clone https://github.com/LMCache/LMCache
cd LMCache
pip install -e .
cd ..
```

(Optional) Install LMCache server if you want to use it as the storage backend
```bash
# (Optional) clone and install LMCache-server if you want to use it as the backend
git clone https://github.com/LMCache/lmcache-server
cd lmcache-server
pip install -e .
cd ..
```

Install vLLM with our integration code
```bash
# clone our modified version of vLLM
git clone https://github.com/LMCache/vllm   
cd vllm

# checkout the dev branch
git checkout dev/lmcache-integration        

# install it to your environment
pip install -e .
```

(Optional) Start redis or LMCache server as the storage backend
```bash
# start LMCache server backend
python3 -m lmcache_server.server localhost 65432

# start redis server 
<you command to start the redis server> # Assuming the server is listening at localhost:6379
```

Run vLLM openai api server 
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model <Your model> \
    --port 8000 \
    --gpu-memory-utilization 0.7 \
    --enable-lmcache \
    --lmcache-local-cache cuda \
    --lmcache-remote-cache lm://localhost:65432 \
    --lmcache-chunksize 256
```

Run the example code in LMCache repo to query the vLLM's openai api server
```bash
cd LMCache/examples
python3 openi_chat_completion_client.py 8000
```

### Docker deployment

We also support docker deployment of vLLM openai api server and LMCache server. Follow the instructions
to build the docker image by yourself.

In the vLLM repo (the `dev/lmcahe-integration` branch)
```bash
DOCKER_BUILDKIT=1 docker build . --target vllm-lmcache --tag vllm-lmcache:latest \
    --build-arg max_jobs=32 --build-arg nvcc_threads=32 --platform linux/amd64
```

In the LMCache-server repo
```bash
docker build . --tag lmcache-server:latest --build-arg max_jobs=32 --platform linux/amd64
```
