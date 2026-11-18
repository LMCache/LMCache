.. _docker_deployment:

Docker deployment
=================

**Prerequisites:** Docker Engine 27.0+

See :ref:`installation_guide` for pulling images.

Running the container
---------------------

Run LMCache as a standalone server and connect the serving engine through its
MP connector. The complete container topology and commands are documented in
:doc:`/mp/deployment`.

The official CUDA vLLM and standalone images run as ``lmcache`` (uid 1000), so
mounted model caches must be readable and writable by that uid. Use
``--volume hf-cache:/home/lmcache/.cache/huggingface`` for a Docker-managed
cache, or ``--user 0`` only when an existing root-owned cache requires it.

ROCm (AMD)
----------

The `AMD Infinity hub <https://hub.docker.com/r/rocm/vllm-dev>`__ for vLLM offers a prebuilt,
optimized image for the AMD Instinct™ MI300X. See
`LLM inference performance validation on AMD Instinct MI300X <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html?model=pyt_vllm_llama-3.1-8b>`__
for full instructions.

Validated environment: ``rocm/vllm-dev:nightly_0624_rc2_0624_rc2_20250620``, MI300X, vLLM V1.

.. code-block:: bash

    docker run -it \
        --network=host \
        --group-add=video \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --device /dev/kfd \
        --device /dev/dri \
        -v <path_to_your_models>:/app/model \
        -e HF_HOME="/app/model" \
        --name lmcache_rocm \
        rocm/vllm-dev:nightly_0624_rc2_0624_rc2_20250620 \
        bash

XPU (Intel)
-----------

The `Intel vLLM XPU hub <https://hub.docker.com/r/intel/vllm>`__ offers a prebuilt, optimized docker image designed for validating inference performance on the Intel GPU accelerators like ARC770, B60/B70, and future products.

Validated environment: ``intel/vllm:0.17.0-xpu``, Intel B60 GPU, vLLM V1.

.. code-block:: bash

    docker run --privileged \
        -it --rm --name vllm-xpu \
        -u root \
        --ipc=host --net=host \
        --cap-add=ALL \
        --device /dev/dri:/dev/dri \
        -v /dev/dri/by-path:/dev/dri/by-path \
        --entrypoint /bin/bash intel/vllm:0.17.0-xpu
