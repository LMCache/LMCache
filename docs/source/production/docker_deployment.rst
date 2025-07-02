.. _docker_deployment:

Docker deployment
=================

Running the container image
---------------------------

You can run the LMCache integrated with vLLM image using Docker as follows:

.. code-block:: bash

    IMAGE=<IMAGE_NAME>:<TAG>
    docker run --runtime nvidia --gpus all \
        --env "HF_TOKEN=<REPLACE_WITH_YOUR_HF_TOKEN>" \
        --env "LMCACHE_CHUNK_SIZE=256" \
        --env "LMCACHE_LOCAL_CPU=True" \
        --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
        --volume ~/.cache/huggingface:/root/.cache/huggingface \
        --network host \
        $IMAGE \
        meta-llama/Llama-3.1-8B-Instruct --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'


The image name and tag can be found on DockerHub - `LMCache/vllm-openai <https://hub.docker.com/r/lmcache/vllm-openai>`_.
See example run file in `docker <https://github.com/LMCache/LMCache/tree/dev/docker>`_ for more details.

.. note::
    DockerHub contains the following image types:
    - Nightly build images of LMCache and vLLM latest code
    - Images of stable releases of LMCache and vLLM