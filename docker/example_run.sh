IMAGE=<IMAGE_NAME>:<TAG>
docker run --runtime nvidia --gpus all \
    --env "HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>" \
    --env "LMCACHE_USE_EXPERIMENTAL=True" \
    --env "chunk_size=256" \
    --env "local_cpu=True" \
    --env "max_local_cpu_size=5" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --network host \
    --entrypoint "/usr/local/bin/vllm" \
    $IMAGE \
    serve mistralai/Mistral-7B-Instruct-v0.2 --kv-transfer-config \
    '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}' \
    --enable-chunked-prefill false
