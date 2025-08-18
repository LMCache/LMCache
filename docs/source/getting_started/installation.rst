.. _installation_guide:

Installation
============

Setup using Python
------------------

Prerequisites
~~~~~~~~~~~~~

- OS: Linux
- Python: 3.10 -- 3.12
- GPU: NVIDIA compute capability 7.0+ (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)
- CUDA 12.8+

.. note::
    LMCache does not support Windows natively. To run LMCache on Windows, you can use the Windows Subsystem for Linux (WSL) with a compatible Linux distribution, or use some community-maintained forks.

Install Stable LMCache from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install the latest stable release of LMCache is through PyPI. If other dependencies
demand a version of torch that differs across major versions (e.g. 2.7.1 versus 2.8.0), LMCache stays compatible
through installation from source (see below). The LMCache is always build with the latest version of torch. 
Installing from source allows torch version flexibility. 

.. code-block:: bash
    
    # LMCache wheels are built with the latest version of torch.
    # If your serving engine pins a different version of torch, it will 
    # override the torch version installed by lmcache
    # if these torch versions differ across major versions, ABI compatibility
    # may break so please install from source (see below)
    pip install lmcache

Install Latest LMCache from TestPyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These wheels are continually built from the latest LMCache source code (not officially stable release). 
If other dependencies demand a version of torch that differs across major versions (e.g. 2.7.1 versus 2.8.0), 
LMCache stays compatible through installation from source (see below). The LMCache torch version always 
matches the latest nightly version of vllm. Installing from source allows torch version flexibility. 

.. code-block:: bash

    # by default, this will port the version of torch of the latest *NIGHTLY* vllm wheel
    # if your serving engine demands a different version of torch, it will 
    # override the torch version installed by lmcache
    # if these torch versions differ across major versions, ABI compatibility
    # may break so please install from source (see below)
    pip install --index-url https://pypi.org/simple --extra-index-url https://test.pypi.org/simple lmcache==0.3.4.dev61

See the latest pre-release of LMCache: `latest LMCache pre-releases <https://test.pypi.org/project/lmcache/#history>`__ and replace `0.3.4.dev61` with the latest pre-release version.

This will install all dependencies from the real PyPI and only LMCache itself from TestPyPI.

Confirm that you have the latest pre-release:

.. code-block:: bash

    python
    >>> import lmcache
    >>> from importlib.metadata import version
    >>> print(version("lmcache"))
    0.3.4.dev61 # should be the latest pre-release version you installed

Install Latest LMCache from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install from source, clone the repository and install in editable mode. 

The reason that torch installation is separated is:
1. different serving engines and different versions of those serving engines 
have different torch dependencies and we want to maintain flexibility 
(torch versions only break across major versions where the ABI may change e.g. 2.7.1 -> 2.8.0). 
2. no build isolation bypasses `PEP 517 <https://peps.python.org/pep-0517/>`_ / `PEP 518 <https://peps.python.org/pep-0518/>`_
avoiding the case where LMCache GPU kernels are compiled with `torch.utils.cuda_extension` or `torch.utils.hipify`
inside of `setup.py` with one torch version while runtime dependencies (unpinned torch version in `requirements/common.txt`)
are overridden, causing undefined symbol references. This forces LMCache to be built with the torch version already in your
environment.

.. code-block:: bash

    git clone https://github.com/LMCache/LMCache.git
    cd LMCache

    # we need to install these packages because we are avoiding build isolation
    pip install -r requirements/build.txt

    # Option 1. 
    # select the torch version that matches the dependency of your serving engine
    # 2.7.1 is an example for vllm 0.10.0
    pip install torch==2.7.1

    # Option 2. 
    # install your serving engine with its required torch version declared already
    # example: vllm 0.10.0 will install torch 2.7.1
    pip install vllm==0.10.0

    # no build isolation requires torch to already be installed
    # with your desired version
    pip install -e . --no-build-isolation

Install LMCache with uv
~~~~~~~~~~~~~~~~~~~~~~~~

We recommend developers to use `uv` for faster package management:

.. code-block:: bash

    git clone https://github.com/LMCache/LMCache.git
    cd LMCache

    uv venv --python 3.12
    source .venv/bin/activate

    # we need to install these packages because we are avoiding build isolation
    uv pip install -r requirements/build.txt

    # Option 1. 
    # select the torch version that matches the dependency of your serving engine
    # 2.7.1 is an example for vllm 0.10.0
    uv pip install torch==2.7.1

    # Option 2. 
    # install your serving engine with its required torch version declared already
    # example: vllm 0.10.0 will install torch 2.7.1
    uv pip install vllm==0.10.0

    # no build isolation requires torch to already be installed
    # with your desired version
    uv pip install -e . --no-build-isolation


LMCache with vLLM v1
~~~~~~~~~~~~~~~~~~~~

LMCache is integrated with the latest vLLM (vLLM v1). To use it, install the latest vLLM package:

.. code-block:: bash

    pip install vllm

Test whether LMCache works with vLLM v1 by running:

.. code-block:: bash

    python3 -c "import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"

LMCache with vLLM v0
~~~~~~~~~~~~~~~~~~~~

.. note::
    LMCache is also integrated with vLLM v0. Refer to `the example in vLLM <https://github.com/vllm-project/vllm/blob/main/examples/others/lmcache/cpu_offload_lmcache.py>`__.
    See the `examples README <https://github.com/vllm-project/vllm/tree/main/examples/others/lmcache#2-cpu-offload-examples>`_ to understand how to run the script for vLLM v0.

Setup using Docker
------------------

Prerequisites
~~~~~~~~~~~~~

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
