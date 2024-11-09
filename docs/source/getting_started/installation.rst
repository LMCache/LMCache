
.. _installation:

Pip Installation
==================

LMCache is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries.

Requirements
------------

* OS: Linux
* Python: 3.10 or higher
* CUDA: 12.1

Install released versions
--------------------------

You can install LMCache using pip:

.. code-block:: console

    $ # (Recommended) Create a new conda environment.
    $ conda create -n venv python=3.10 -y
    $ conda activate venv

    $ # Install vLLM with CUDA 12.1.
    $ pip install lmcache lmcache_vllm

.. note::

    Although we recommend using ``conda`` to create and manage Python environments, it is highly recommended to use ``pip`` to install LMCache. This is because ``pip`` can install ``torch`` with separate library packages like ``NCCL``, while ``conda`` installs ``torch`` with statically linked ``NCCL``. This can cause issues when vLLM tries to use ``NCCL``.
    As LMCache depends on vLLM as a backend, it is necessary to install vLLM correctly.

.. note::

    LMCache provides the integration to the latest vLLM (0.6.2)

.. note::
    LMCache requires CUDA 12.1. You can check ``nvcc --version`` to see if you loaded CUDA 12. Following, please add the following to your ``~/.bashrc`` file:

.. code-block:: bash
    
    cuda_version=12.1
    export CUDA_HOME=/usr/local/cuda-${cuda_version}
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH

Install the latest code
----------------------------

You can install the latest code from the GitHub repository:

.. code-block:: console

    # Install vLLM==0.6.2
    $ pip install vllm==0.6.2

    # Clone and install LMCache
    $ git clone git@github.com:LMCache/LMCache.git
    $ cd LMCache
    $ pip install -e .
    $ cd ..

    # Clone and install LMCache-vLLM
    $ git clone git@github:LMCache/lmcache-vllm.git
    $ cd lmcache-vllm
    $ pip install -e .
    $ cd ..

Quickstart
----------

LMCache has the same interface as vLLM (both online serving and offline inference). 
To use the online serving, you can start an OpenAI API-compatible vLLM server with LMCache via:

.. code-block:: console

    $ lmcache_vllm serve lmsys/longchat-7b-16k --gpu-memory-utilization 0.8

To use vLLM's offline inference with LMCache, just simply add ``lmcache_vllm`` before the import to the vLLM components. For example

.. code-block:: python

    import lmcache_vllm.vllm as vllm
    from lmcache_vllm.vllm import LLM 

    # Load the model
    model = LLM.from_pretrained("lmsys/longchat-7b-16k")

    # Use the model
    model.generate("Hello, my name is", max_length=100)




