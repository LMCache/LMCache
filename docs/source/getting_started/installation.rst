
.. _installation:

Pip Installation
==================

LMCache is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries.

Requirements
------------

* OS: Linux
* Python: 3.10 or higher
* CUDA: 12.1

Install pip released versions (v0)
-----------------------------------

You can install LMCache using pip:

.. code-block:: console

    $ # (Recommended) Create a new conda environment.
    $ conda create -n venv python=3.10 -y
    $ conda activate venv

    $ # Install vLLM with CUDA 12.1.
    $ pip install lmcache==0.1.4 lmcache_vllm==0.6.2.3

.. note::

    Although we recommend using ``conda`` to create and manage Python environments, it is highly recommended to use ``pip`` to install LMCache. This is because ``pip`` can install ``torch`` with separate library packages like ``NCCL``, while ``conda`` installs ``torch`` with statically linked ``NCCL``. This can cause issues when vLLM tries to use ``NCCL``.
    As LMCache depends on vLLM as a backend, it is necessary to install vLLM correctly.

.. note::

    pip install for LMCache v1 is not available yet (will be released soon). 
    Please install LMCache v1 from source for now.

.. note::
    LMCache requires CUDA 12.1. You can check ``nvcc --version`` to see if you loaded CUDA 12. Following, please add the following to your ``~/.bashrc`` file:

.. code-block:: bash
    
    cuda_version=12.1
    export CUDA_HOME=/usr/local/cuda-${cuda_version}
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_HOME/bin:$PATH



Install from source (v1)
----------------------------

You can install the latest code from the GitHub repository:

.. code-block:: console

    # vLLM version: 0.7.4.dev160+g28943d36
    # NOTE: Run the below script in a virtual environment to avoid mess up the default env
    $ pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly
    $ git clone https://github.com/LMCache/LMCache.git
    $ cd LMCache 
    $ pip install -e .


Install from source (v0)
----------------------------

You can install the latest code from the GitHub repository:

.. code-block:: console

    # Install vLLM version
    $ pip install vllm==0.6.2.3

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

Version Compatibility Matrix
------------------------------

+--------------------+------------------------+---------------+
| LMCache            | LMCache_vLLM           | vLLM          |
+--------------------+------------------------+---------------+
| v1                 |     N/A                | 0.7.3         |
+--------------------+------------------------+---------------+
| 0.1.4 (v0)         | 0.6.2.3                | 0.6.2         |
+--------------------+------------------------+---------------+
| 0.1.3 (v0)         | 0.6.2.2                | 0.6.1.post2   |
+--------------------+------------------------+---------------+

.. note::
    For LMCache v1, please refer to the examples in the :ref:`v1_index` section. 
    LMCache v1 can be directly run with the ``vllm serve`` command.

.. note::
    For LMCache v1, LMCACHE_USE_EXPERIMENTAL=True is required to use the experimental features.

Quickstart (v1)
---------------

For LMCache v1, you can start the LMCache server with the following command:

.. code-block:: bash

    LMCACHE_CONFIG_FILE=./lmcache_config.yaml \
    LMCACHE_USE_EXPERIMENTAL=True vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
    --max-model-len 4096  --gpu-memory-utilization 0.8 --port 8000 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnector", "kv_role":"kv_both"}'

Quickstart (v0)
---------------

For LMCache v0, you can start the LMCache server with the following command:

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






