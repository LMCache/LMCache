# LMCache core library

## Installation

**Prerequisite:** Python >= 3.10

```bash
pip install -e .
```

## Quickstart: 

**Prerequisites**: To run the quickstart demo, your server should have 1 GPU and the [docker environment](https://docs.docker.com/engine/install/) installed.

**Step 1:** Pull docker images
```bash
docker pull apostacyh/vllm:lmcache-0.1.0
```

**Step 2:** Start vLLM + LMCache 
```bash
model=mistralai/Mistral-7B-Instruct-v0.2    # Replace with your model name
sudo docker run --runtime nvidia --gpus '"device=0"' \
    -v <Huggingface cache dir on your local machine>:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<Your huggingface access token>" \
    --ipc=host \
    --network=host \
    apostacyh/vllm:lmcache-0.1.0 \
    --model $model --gpu-memory-utilization 0.6 --port 8000 \
    --lmcache-config-file /lmcache/LMCache/examples/example-local.yaml
```
Please fill `Huggingface cache dir on your local machine` and `Your huggingface access token` in the above command. 

You can also change the `model` variable to use different models.

The vLLM engine is ready after you see the logs like this:
```
INFO:     Started server process [865615]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Step 3:** Run demo application

You can now run the demo application in the LMCache repo. Please execute the following commands on the server
```bash
git clone https://github.com/LMCache/LMCache
cd LMCache/examples/

# Install openai client library
pip install openai

# Start the demo chat application
python openai_chat_completion_client.py 8000
```

This demo is a QA application based on a long context (`examples/f.txt`). The TTFT should be drastically reduced since the second round of QA.



## Use case 1: share prefix KV between different vLLM instance through LMCache
The following instructions help deploy LMCache backend + multile vLLM instance by docker containers. The architecture of the demo application looks like this:

<img width="817" alt="image" src="https://github.com/LMCache/LMCache/assets/25103655/ab64f84d-26e1-46ce-a503-e7e917b618bc">


**Prerequisites**: To run the quickstart demo, your server must have 2 GPUs and the [docker environment](https://docs.docker.com/engine/install/) installed.


**Step 1:** Pull docker images
```bash
docker pull apostacyh/lmcache-server:0.1.0
docker pull apostacyh/vllm:lmcache-0.1.0
```

**Step 2:** Start LMCache backend server 
```bash
docker run --name apostacyh/lmcache-server:0.1.0 --network host -d lmcache-server:latest 0.0.0.0 65432
```

**Step 3:** start 2 vLLM instances
```bash
# The first vLLM instance listens at port 8000
model=mistralai/Mistral-7B-Instruct-v0.2    # Replace with your model name
sudo docker run --runtime nvidia --gpus '"device=0"' \
    -v <Huggingface cache dir on your local machine>:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=<Your huggingface token>" \
    --ipc=host \
    --network=host \
    apostacyh/vllm:lmcache-0.1.0 \
    --model $model --gpu-memory-utilization 0.7 --port 8000 \
    --lmcache-config-file /lmcache/LMCache/examples/example.yaml
```

Now, open another terminal and start another vLLM instance
```bash
# The second vLLM instance listens at port 8001
model=mistralai/Mistral-7B-Instruct-v0.2    # Replace with your model name
sudo docker run --runtime nvidia --gpus '"device=1"' \
    -v <Huggingface cache dir on your local machine>:/root/.cache/huggingface \
    -p 8001:8001 \
    --env "HF_TOKEN=<Your huggingface token>" \
    --ipc=host \
    --network=host \
    apostacyh/vllm:lmcache-0.1.0 \
    --model $model --gpu-memory-utilization 0.7 --port 8001 \
    --lmcache-config-file /lmcache/LMCache/examples/example.yaml
```

Remember to replace the `Huggingface cache dir on your local machine` and `Your huggingface token` in the commandline.

The vLLM engines are ready after you see the logs like this:
```
INFO:     Started server process [865615]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Step 4:** Run demo application
You can run the demo application in the LMCache repo. Please execute the following commands on the server
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

