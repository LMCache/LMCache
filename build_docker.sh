lmcache_version_id=$(pip index versions lmcache | grep "Available" | awk '{print $3}')
DOCKER_BUILDKIT=1 docker build \
    --build-arg LMCACHE_VERSION=$lmcache_version . \
    --target vllm-lmcache \
    --tag vllm-lmcache:test \
    --build-arg max_jobs=32 \
    --build-arg nvcc_threads=32 \
    --platform linux/amd64

