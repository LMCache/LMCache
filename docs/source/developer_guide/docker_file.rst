Dockerfile
==========

We provide a Dockerfile to help you build a Docker image for LMCache. More information about deploying LMCache with Docker can be 
found here - :ref:`Docker deployment guide <docker_deployment>`.

Example run command
-------------------

.. code-block:: bash

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


The Image Name and Tag can be found on Docker Hub - `LMCache <https://hub.docker.com/r/lmcache/vllm-openai>`_.

Building the Docker image
-------------------------

You can build and run LMCache with Docker from source via the provided Dockerfile.
The Dockerfile is located in at `Dockerfile <https://github.com/LMCache/LMCache/tree/dev/docker>`_.

To build the Docker image, run the following command from the root directory of the LMCache repository:

.. code-block:: bash

    docker build -t <IMAGE_NAME>:<TAG> -f docker/Dockerfile .

Replace `<IMAGE_NAME>` and `<TAG>` with your desired image name and tag.





