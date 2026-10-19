.. _docker_deployment:

Docker deployment
=================

.. warning::

   This page documents the behavior of LMCache's in-process mode (deprecated). Please consider using :doc:`LMCache MP mode </mp/index>` for better feature support and performance. For the related MP documentation and its separate configuration, see :doc:`/mp/deployment`.


**Prerequisites:** Docker Engine 27.0+

See :ref:`installation_guide` for pulling images.

Running the container
---------------------

.. code-block:: bash

    IMAGE=<IMAGE_NAME>:<TAG>
    mkdir -p "$HOME/.cache/huggingface"
    docker run --runtime nvidia --gpus all \
        --env "HF_TOKEN=<REPLACE_WITH_YOUR_HF_TOKEN>" \
        --env "LMCACHE_CHUNK_SIZE=256" \
        --env "LMCACHE_LOCAL_CPU=True" \
        --env "LMCACHE_MAX_LOCAL_CPU_SIZE=5" \
        --volume "$HOME/.cache/huggingface:/home/lmcache/.cache/huggingface" \
        --network host \
        $IMAGE \
        meta-llama/Llama-3.1-8B-Instruct --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

See the `docker run example <https://github.com/LMCache/LMCache/tree/dev/docker>`_ for more details.

For container users, Hugging Face cache ownership, and other accelerator
images, see the shared :doc:`/production/docker_deployment` and
:doc:`/getting_started/installation` guides.
