.. _installation_guide:

Installation
============

**Prerequisites:** Linux · Python 3.9–3.13 · NVIDIA GPU (compute 7.0+) · CUDA 12.1+ · `uv <https://astral.sh/uv>`_

Install LMCache
---------------

.. tab-set::

    .. tab-item:: Python (pip / uv)

        .. tab-set::

            .. tab-item:: Stable  

                .. tab-set::

                    .. tab-item:: CUDA 12.9  

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache

                        .. important::

                            You're all set! You can now start using LMCache. For hands-on guides and more
                            usage examples, see the :ref:`quickstart_examples` section.

                    .. tab-item:: CUDA 13.0

                        The CUDA 13.0 wheel is published to a dedicated
                        `GitHub Release <https://github.com/LMCache/LMCache/releases>`__ rather than PyPI.

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            VERSION=0.4.3  # replace with target release
                            uv pip install lmcache==${VERSION} \
                                --extra-index-url https://download.pytorch.org/whl/cu130 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/v${VERSION}-cu13 \
                                --index-strategy unsafe-best-match

                        .. note::

                            ``--extra-index-url https://download.pytorch.org/whl/cu130`` ensures the CUDA 13.0
                            build of PyTorch is resolved. Without it, pip may select a mismatched CUDA variant.

            .. tab-item:: Nightly

                Nightly wheels are built from the latest ``dev`` branch each day at 07:30 UTC
                and published to GitHub Releases. No version pinning required — ``--pre``
                picks the latest nightly automatically.

                .. tab-set::

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache --pre \
                                --extra-index-url https://download.pytorch.org/whl/cu129 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/nightly \
                                --index-strategy unsafe-best-match

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            uv venv --python 3.12
                            source .venv/bin/activate
                            uv pip install lmcache --pre \
                                --extra-index-url https://download.pytorch.org/whl/cu130 \
                                --find-links https://github.com/LMCache/LMCache/releases/expanded_assets/nightly-cu13 \
                                --index-strategy unsafe-best-match

            .. tab-item:: From Source

                ``--no-build-isolation`` ensures the kernels are compiled against the same torch
                already installed in your environment, preventing undefined symbol errors at runtime.

                .. tab-set::

                    .. tab-item:: CUDA

                        .. code-block:: bash

                            git clone https://github.com/LMCache/LMCache.git
                            cd LMCache

                            uv venv --python 3.12
                            source .venv/bin/activate

                            uv pip install -r requirements/build.txt
                            uv pip install vllm  # pulls in required torch version
                            uv pip install -e . --no-build-isolation

                    .. tab-item:: ROCm

                        .. code-block:: bash

                            git clone https://github.com/LMCache/LMCache.git
                            cd LMCache

                            uv venv --python 3.12
                            source .venv/bin/activate

                            # Need to install these packages manually to avoid build isolation
                            uv pip install -r requirements/build.txt

                            # Install torch from the ROCm wheel index
                            uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0

                            # Build LMCache. BUILD_WITH_HIP=1 makes setup.py pick cupy-rocm-7-0 automatically.
                            PYTORCH_ROCM_ARCH="gfx942" \
                            TORCH_DONT_CHECK_COMPILER_ABI=1 \
                            CXX=hipcc \
                            BUILD_WITH_HIP=1 \
                            uv pip install -e . --no-build-isolation

    .. tab-item:: Docker

        .. tab-set::

            .. tab-item:: Stable

                .. tab-set::

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-cu13

            .. tab-item:: Nightly

                .. tab-set::

                    .. tab-item:: CUDA 12.9

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-nightly

                    .. tab-item:: CUDA 13.0

                        .. code-block:: bash

                            docker pull lmcache/vllm-openai:latest-nightly-cu13

            .. tab-item:: ROCm

                .. code-block:: bash

                    docker pull rocm/vllm-dev:nightly_0624_rc2_0624_rc2_20250620

        See :ref:`docker_deployment` for running the container and ROCm images.

    .. tab-item:: CLI Only  

        Lightweight CLI-only package for querying or benchmarking a remote LMCache server.
        No CUDA required, works on any OS.

        .. code-block:: bash

            pip install lmcache-cli

        .. note::

            ``lmcache-cli`` and ``lmcache`` ship the same ``lmcache`` CLI command.
            Do not install both in the same environment.

Verify Installation
-------------------

.. code-block:: bash

    python -c "import lmcache.c_ops"

Compatibility Matrix
~~~~~~~~~~~~~~~~~~~~

✅ compatible · ❌ API incompatible · 🕯️ torch mismatch (use ``--no-build-isolation``)


.. container:: compat-table-scroll

   .. csv-table::
      :file: Installation_compatibility_matrix.csv
      :header-rows: 1

Notable Change List: 

* June 30: vLLM Cached Req Scheduler Output Changes https://github.com/vllm-project/vllm/pull/20232 and https://github.com/vllm-project/vllm/pull/20291

Setup using Docker
------------------

Docker Prerequisites
~~~~~~~~~~~~~~~~~~~~

- Docker Engine 27.0+

Pre-built LMCache integrated with vLLM Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide pre-built container images of LMCache integrated with vLLM.

You can get the latest stable image as follows:

.. code-block:: bash

    docker pull lmcache/vllm-openai

You can get the nightly build of latest code of LMcache and vLLM as follows:

.. code-block:: bash

    docker pull lmcache/vllm-openai:latest-nightly


LMCache on ROCm
------------------

Get started through using vLLM docker image as base image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `AMD Infinity hub <https://hub.docker.com/r/rocm/vllm-dev>`__ for vLLM offers a prebuilt, optimized docker image designed for validating inference performance on the AMD Instinct™ MI300X accelerator.
The image is based on the latest vLLM v1. Please check `LLM inference performance validation on AMD Instinct MI300X <https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/benchmark-docker/vllm.html?model=pyt_vllm_llama-3.1-8b>`__ for instructions on how to use this prebuilt docker image.

As of the date of writing, the steps are validated on the following environment:

- docker image: rocm/vllm-dev:nightly_0624_rc2_0624_rc2_20250620
- MI300X
- vLLM V1

.. code-block:: bash

    #!/bin/bash
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

Install Latest LMCache from Source for ROCm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install from source, clone the repository and install in editable mode.

.. code-block:: bash

    PYTORCH_ROCM_ARCH="{your_rocm_arch}" \
    TORCH_DONT_CHECK_COMPILER_ABI=1 \
    CXX=hipcc \
    BUILD_WITH_HIP=1 \
    python3 -m pip install --no-build-isolation -e .

Example on MI300X (gfx942):

.. code-block:: bash

    PYTORCH_ROCM_ARCH="gfx942" \
    TORCH_DONT_CHECK_COMPILER_ABI=1 \
    CXX=hipcc \
    BUILD_WITH_HIP=1 \
    python3 -m pip install --no-build-isolation -e .


LMCache on Intel XPU
------------------

Get started through using vLLM docker image as base image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Intel vLLM XPU hub<https://hub.docker.com/r/intel/vllm>`__ offers a prebuilt, optimized docker image designed for validating inference performance on the Intel GPU accelerator like PVC, BMG, and future products.

User could also build latest dev image by following the instructions below:

.. code-block:: bash

    git clone https://github.com/vllm-project/vllm.git
    cd vllm
    docker build --network=host -t vllm-xpu:dev --file docker/Dockerfile.xpu .
    docker run --privileged -it --rm --name vllm-xpu -u root --ipc=host --net=host --cap-add=ALL --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path --entrypoint /bin/bash vllm-xpu:dev

Install Latest LMCache from Source for Intel XPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install from source, clone the repository and install in editable mode.

.. code-block:: bash

   BUILD_WITH_SYCL=1 pip install --no-build-isolation -e .

