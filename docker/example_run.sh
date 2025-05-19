# Example script to run the container vLLM OpenAI server with LMCache
#
# Prerequisite:
# - If CUDA then require NVIDIA Container Toolkit:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Set the following variables:
IMAGE=
HF_MODEL_NAME='meta-llama/Llama-3.1-8B-Instruct'
RUNTIME=nvidia

docker run --runtime=$RUNTIME --gpus all \
    --env "HF_TOKEN=<HF_TOKEN_ACCESS_MODEL>" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "chunk_size=256" \
    --env "local_cpu=True" \
    --env "max_local_cpu_size=5" \
    --volume ~/.cache/huggingface:/root/.cache/huggingface \
    --network host \
    $IMAGE \
    $HF_MODEL_NAME --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
