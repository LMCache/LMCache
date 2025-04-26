Installation
============

Prerequisites
-------------

- Python 3.10 - 3.12
- CUDA 12.4 (minimal)
- PyTorch 2.6.0

Setup using Python
------------------

Install LMCache from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install LMCache is through PyPI:

.. code-block:: bash

    pip install lmcache

Install LMCache from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install from source, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/LMCache/LMCache.git
    cd LMCache
    pip install -e .

Make LMCache work with latest vLLM v1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LMCache is integrated with latest vLLM v1. To use it, you need to install the latest vLLM main branch:

.. code-block:: bash

    pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly


Then, install a small patch to enable vLLM v1 use LMCache:

.. code-block:: bash

    git clone https://github.com/LMCache/LMCache.git
    cd LMCache/contrib
    python3 install_modules.py

.. note::
    This patch will no longer need to be installed after PR `vllm-project/vllm#16625 <https://github.com/vllm-project/vllm/pull/16625>`_ is merged.


Test whether LMCache works with vLLM v1:

.. code-block:: bash

    python3 -c "import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"


Setup using Docker
------------------

Pre-built vLLM + LMCache Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide pre-built Docker images that include vLLM integration:

.. code-block:: bash

    docker pull lmcache/vllm-openai:2025-04-18
    
.. note::
    Currently, we build and release Docker images manually. An automated Docker build/release GitHub workflow will be set up soon. Contributions to this effort are welcomed!
