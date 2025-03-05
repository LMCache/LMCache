.. _docker:

Docker Installation
=========================

LMCache offers an official Docker image for deployment (for LMCache v1). 
The image is available on Docker Hub at `lmcache/vllm-openai <https://hub.docker.com/r/lmcache/vllm-openai>`_ .


.. note::

    Make sure you have Docker installed on your machine. You can install Docker from `here <https://docs.docker.com/get-docker/>`_.

.. note::

    The Docker image `lmcache/lmcache_vllm <https://hub.docker.com/r/lmcache/lmcache_vllm>`_ 
    for LMCache v0 is no longer maintained.

Pulling the Docker Image:
----------------------------

To get started, pull the official Docker image with the following command:

.. code-block:: bash

    docker pull lmcache/vllm-openai

Running the Docker Container
---------------------------------------


1. Run the Docker Command:

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

Save the above command in a file named ``run.sh`` and run the following command:

.. code-block:: bash

    chmod +x run.sh
    ./run.sh


Testing the Docker Container
--------------------------------

To verify the setup, you can test it using the following ``curl`` command:

.. code-block:: bash

    curl -X 'POST' \
    'http://127.0.0.1:8000/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
        "model": "meta-llama/Llama-3.2-1B",
        "messages": [
        {"role": "system", "content": "You are a helpful AI coding assistant."},
        {"role": "user", "content": "Write a segment tree implementation in python"}
        ],
        "max_tokens": 150
    }'


Building Docker from Source
----------------------------

.. note::

    This section is for users who want to build the Docker image from source.
    For this please visit the link here `LMCache docker <https://github.com/LMCache/LMCache/tree/dev/docker>`_.

    
