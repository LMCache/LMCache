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

.. code-block:: bash

    docker pull lmcache/lmcache_vllm:lmcache-0.1.3

Running the Docker Container
---------------------------------------

To run the Docker container with your specified model, follow these steps:

1. Define the Model:

.. code-block:: bash

    # define the model here
    export model=meta-llama/Llama-3.2-1B

2. Create Configuration and Chat Template Files

Save the following YAML code to a file, such as ``example.yaml``, in the LMCache repository:

.. code-block:: yaml
    
    chunk_size: 256
    local_device: "cpu"

    # Whether retrieve() is pipelined or not
    pipelined_backend: False

Save the chat template code below to a file, such as ``chat-template.txt``, in the LMCache repository:

.. code-block:: text

    {%- if messages[0]['role'] == 'system' -%}  
        {%- set system_message = messages[0]['content'] -%}    
        {%- set messages = messages[1:] -%}
    {%- else -%}    
        {% set system_message = '' -%}{%- endif -%}
    {{ bos_token + system_message }}
    {%- for message in messages -%}
        {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
            {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        {%- endif -%}    
        {%- if message['role'] == 'user' -%}        
            {{ 'USER: ' + message['content'] + '\n' }}    
        {%- elif message['role'] == 'assistant' -%}        
            {{ 'ASSISTANT: ' + message['content'] + eos_token + '\n' }}    
        {%- endif -%}
    {%- endfor -%}
    {%- if add_generation_prompt -%}   
        {{ 'ASSISTANT:' }} 
    {% endif %}

3. Run the Docker Command:

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

.. code-block:: bash

    # Coming soon