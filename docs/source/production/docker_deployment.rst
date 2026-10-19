Docker Images and Runtime Setup
===============================

Image selection, device access, and model-cache permissions are shared
prerequisites. Choose a launch command for the intended LMCache mode:

- :doc:`/mp/deployment` starts the standalone MP server and inference worker.
  It also documents container users, model-cache permissions, and IPC setup.
- :doc:`/legacy/production/docker_deployment` contains the deprecated
  ``LMCacheConnectorV1`` embedded launch and engine YAML environment settings.
- :doc:`/getting_started/installation` lists installation channels and images.

The accelerator shell examples below record the environments originally used
to check container access. They do not establish MP connector compatibility
for those historical image versions. Install matching LMCache and engine
packages and use the MP launch/configuration guide for current deployments.

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
