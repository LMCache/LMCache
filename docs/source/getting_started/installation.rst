.. _installation_guide:

Installation
============

Prerequisites
-------------

- Python 3.10+
- CUDA 12.4+

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

Install LMCache with uv
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend developers to use `uv` for a better package management:

.. code-block:: bash

    git clone https://github.com/LMCache/LMCache.git
    cd LMCache
    
    uv venv --python 3.12
    source .venv/bin/activate
    uv pip install -e .


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
    LMCache is also integrated with vLLM v0. Refer to `the example in vLLM <https://github.com/vllm-project/vllm/blob/main/examples/lmcache/cpu_offload_lmcache.py>`__.
    See the `examples README <https://github.com/vllm-project/vllm/tree/main/examples/lmcache#2-cpu-offload-examples>`_ to understand how to run the script for vLLM v0.

Setup using Docker
------------------

Pre-built vLLM + LMCache Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide pre-built Docker images that include vLLM integration:

.. code-block:: bash

    docker pull lmcache/vllm-openai:2025-04-18
    
.. note::
    Currently, we build and release Docker images manually. An automated Docker build/release GitHub workflow will be set up soon. Contributions to this effort are welcomed!
