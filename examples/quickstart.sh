#!/bin/bash

### CHANGE THE FOLLOWING VARIABLES BASED ON YOUR SETTING
MODEL=lmsys/longchat-7b-16k                 # LLM model name
LOCAL_HF_HOME=${HOME}/.cache/huggingface/   # the HF_HOME on local machine. vLLM will try finding/dowloading the models here
HF_TOKEN=                                   # (optional) the huggingface token to access some special models
PORT=8000                                   # Port for the server

sudo docker pull apostacyh/vllm:lmcache-0.1.0
sudo docker run --runtime nvidia --gpus '"device=0"' \
    -v ${LOCAL_HF_HOME}:/root/.cache/huggingface \
    --env "HF_TOKEN=${HF_TOKEN}" \
    --ipc=host \
    --network=host \
    apostacyh/vllm:lmcache-0.1.0 \
    --model ${MODEL} --gpu-memory-utilization 0.6 --port ${PORT} \
    --lmcache-config-file /lmcache/LMCache/examples/example-local.yaml
