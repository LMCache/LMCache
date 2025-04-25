Installation
============

Prerequisites
-------------

- Python 3.10 - 3.12
- CUDA 12.4 (minimal)
- PyTorch 2.6.0

Setup using Python
------------------

Install from PyPI
~~~~~~~~~~~~~~~~~

The simplest way to install LMCache is through PyPI:

.. code-block:: bash

    pip install lmcache

Install from Source
~~~~~~~~~~~~~~~~~~~

To install from source, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/LMCache/LMCache.git
    cd LMCache
    pip install -e .

Setup using Docker
------------------

Pre-built vLLM + LMCache Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide pre-built Docker images that include vLLM integration:

.. code-block:: bash

    docker pull lmcache/vllm-openai:2025-04-18
    
.. note::
    Currently, we build and release Docker images manually. An automated Docker build/release GitHub workflow will be set up soon. Contributions to this effort are welcomed!
