.. _installation_guide:

Installation
============

Setup using Python
------------------

Prerequisites
~~~~~~~~~~~~~

- Python 3.10+
- CUDA 12.8+

Install Stable LMCache from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install the latest stable release of LMCache is through PyPI:

.. code-block:: bash

    pip install lmcache

Install Latest LMCache from TestPyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These wheels are continually built from the latest LMCache source code (not officially stable release).

.. code-block:: bash

    pip install --index-url https://pypi.org/simple --extra-index-url https://test.pypi.org/simple lmcache==0.2.2.dev57

See the latest pre-release of LMCache: `latest LMCache pre-releases <https://test.pypi.org/project/lmcache/#history>`__ and replace `0.2.2.dev57` with the latest pre-release version.

This will install all dependencies from the real PyPI and only LMCache itself from TestPyPI.

Confirm that you have the latest pre-release:

.. code-block:: bash

    python
    >>> import lmcache
    >>> from importlib.metadata import version
    >>> print(version("lmcache"))
    0.2.2.dev57 # should be the latest pre-release version you installed

Install Latest LMCache from Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install from source, clone the repository and install in editable mode:

.. code-block:: bash

    git clone https://github.com/LMCache/LMCache.git
    cd LMCache
    pip install -e .

Install LMCache with uv
~~~~~~~~~~~~~~~~~~~~~~~~

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
