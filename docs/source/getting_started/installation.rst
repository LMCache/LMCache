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

LMCache with vLLM v1
~~~~~~~~~~~~~~~~~~~~

LMCache is integrated with latest vLLM v1. To use it, install the latest vLLM main branch:

.. code-block:: bash

    pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

Test whether LMCache works with vLLM v1 by running:

.. code-block:: bash

    python3 -c "import vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"

LMCache with vLLM v0
~~~~~~~~~~~~~~~~~~~~

.. note::
    LMCache is also integrated with vLLM v0. The documentation is right now work in progress, but you can refer to `this example in vLLM <https://github.com/vllm-project/vllm/blob/main/examples/lmcache/cpu_offload_lmcache_v0.py>`__.

Setup using Docker
------------------

Pre-built vLLM + LMCache Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide pre-built Docker images that include vLLM integration:

.. code-block:: bash

    docker pull lmcache/vllm-openai:2025-04-18
    
.. note::
    Currently, we build and release Docker images manually. An automated Docker build/release GitHub workflow will be set up soon. Contributions to this effort are welcomed!
