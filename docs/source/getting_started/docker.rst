.. _docker:

LMCache with Docker
=========================

LMCache offers an official Docker image for deployment. 
The image is available on Docker Hub at `lmcache/lmcache_vllm <https://hub.docker.com/r/lmcache/lmcache_vllm>`_ .


.. note::

    Make sure you have Docker installed on your machine. You can install Docker from `here <https://docs.docker.com/get-docker/>`_.

Pulling the Docker Image:
----------------------------

To get started, pull the official Docker image with the following command:

.. code-block:: console

    docker pull lmcache/lmcache_vllm:lmcache-0.1.3

Running the Docker Container
---------------------------------------

To run the Docker container with your specified model, follow these steps:

1. Define the Model:

.. code-block:: bash

    # define the model here
    export model=meta-llama/Llama-3.2-1B

2. Run the Docker Command:

.. code-block:: bash

    docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v <Path to LMCache>:/etc/lmcache \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=<Your Huggingface Token>" \
    --env "LMCACHE_CONFIG_FILE=/etc/lmcache/example.yaml"\
    --env "VLLM_WORKER_MULTIPROC_METHOD=spawn"\
    --ipc=host \
    --network=host \
    lmcache/lmcache_vllm:lmcache-0.1.3 \
    $model --gpu-memory-utilization 0.7 --port 8000 \
    --chat_template /etc/lmcache/chat-template.txt 

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

To build and run LMCache from source, use the provided Dockerfile. First, clone the LMCache repository and build the Docker image with the following commands:

.. code-block:: bash

    lmcache_version_id=$(pip index versions lmcache | grep "Available" | awk '{print $3}')
    DOCKER_BUILDKIT=1 docker build \
        --build-arg LMCACHE_VERSION=$lmcache_version . \
        --target vllm-lmcache \
        --tag vllm-lmcache:test \
        --build-arg max_jobs=32 \
        --build-arg nvcc_threads=32 \
        --platform linux/amd64

To run the Docker container, follow the steps in the **Running the Docker Container** section, but replace the image tag ``lmcache/lmcache_vllm:lmcache-0.1.3`` with ``vllm-lmcache:test``.

