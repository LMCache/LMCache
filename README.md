# LMCache core library

## Installation

**Prerequisite:** Python >= 3.10

```bash
pip install -e .
```

## Quickstart
The following instructions help deploy LMCache + vLLM by docker containers. The architecture of the demo application looks like this:

<img width="817" alt="image" src="https://github.com/LMCache/LMCache/assets/25103655/ab64f84d-26e1-46ce-a503-e7e917b618bc">


**Prerequisites**: To run the quickstart demo, your server must have 2 GPUs and the [docker environment](https://docs.docker.com/engine/install/) installed.


**Step 1:** Pull docker images
```bash
docker pull apostacyh/lmcache-server:latest
docker pull apostacyh/vllm:lmcache
```

**Step 2:** Start lmcache server 
```bash
docker run --name apostacyh/lmcache-server --network host -d lmcache-server:latest 0.0.0.0 65432
```

**Step 3:** start 2 vLLM instances
```bash
# The first vLLM instance listens at port 8000
sudo docker run --runtime nvidia --gpus '"device=0"' \
    -v <Hugging face cache dir on your local machine>:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<Your huggingface token>" \
    --ipc=host \
    --network=host \
    apostacyh/vllm:lmcache \
    --model lmsys/longchat-7b-16k --gpu-memory-utilization 0.7 --port 8000 \
    --lmcache-config-file /lmcache/LMCache/examples/example.yaml
```

Now, open another terminal and start another vLLM instance
```bash
# The second vLLM instance listens at port 8001
# The first vLLM instance listens at port 8000
sudo docker run --runtime nvidia --gpus '"device=1"' \
    -v <Hugging face cache dir on your local machine>:/root/.cache/huggingface \
    -p 8001:8001 \
    --env "HF_TOKEN=<Your huggingface token>" \
    --ipc=host \
    --network=host \
    apostacyh/vllm:lmcache \
    --model lmsys/longchat-7b-16k --gpu-memory-utilization 0.7 --port 8001 \
    --lmcache-config-file /lmcache/LMCache/examples/example.yaml
```

The vLLM engines are ready after you see the logs like this:
```
INFO:     Started server process [865615]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Step 4:** Run demo application
You can run the demo application in the LMCache repo. Please execute the following commands on the host device 
```bash
git clone https://github.com/LMCache/LMCache
cd LMCache/examples/

# Install openai client library
pip install openai
```

In one terminal:
```
# Connect to the first vLLM engine
python openai_chat_completion_client.py 8000
```

In another terminal
```
# Connect to the second vLLM engine
python openai_chat_completion_client.py 8001
```

You should be able to see the second vLLM engine has much lower response delay.
This is because the KV cache of the long context can be shared across both vLLM engines by using LMCache.

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

### Local testing (Outdated)

To vLLM + LMCache locally, please follow the following steps:

**NOTE:** We highly recommend running the installation in a newly created python virtual environment. Here are example commands:
```bash
# If you are using conda
conda create -n <your env name> python=3.10
conda activate <your env name>

# Or, if you are using venv (make sure python version >= 3.10)
python -m venv <path to your env>
source <path to you env>/bin/activate
```

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

# Start Redis server 
docker run --name my-redis -p 6379:6379 -d redis
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
