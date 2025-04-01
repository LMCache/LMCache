
.. _installation:

Installation Guide
====================

LMCache is a Python library that also contains pre-compiled C++ and CUDA (12.1) binaries.

Requirements
------------

* OS: Linux
* Python: 3.10 or higher
* CUDA: 12.1

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

.. note::
    For LMCache v1, LMCACHE_USE_EXPERIMENTAL=True is required to use the experimental features. The 
    relevant source code is in the ``lmcache/experimental`` directory in the ``dev`` branch of the 
    LMCache repository. Source installation is the same for v0 and v1 but v0 doesn't require
    LMCACHE_USE_EXPERIMENTAL=True.

.. note::
    For LMCache v1, please refer to the examples in the :ref:`v1_index` section. 
    LMCache v1 can be directly run with the ``vllm serve`` command.

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
