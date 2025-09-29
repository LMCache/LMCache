LMCache Server
==============

.. _lmserver-overview:

Overview
--------

LMCache Server is a built-in remote storage backend that provides a centralized KV cache storage solution. 
The ``lmcache_server`` CLI entrypoint starts a remote LMCache server and comes with the ``lmcache`` package.

Some other remote backends are :doc:`Redis <./redis>`, :doc:`Mooncake <./mooncake>`, :doc:`Valkey <./valkey>`, and :doc:`InfiniStore <./infinistore>`.

Two ways to configure LMCache Server Offloading:
------------------------------------------------

**1. Environment Variables:**

``LMCACHE_USE_EXPERIMENTAL`` MUST be set by environment variable directly.

.. code-block:: bash

    # Specify LMCache V1
    export LMCACHE_USE_EXPERIMENTAL=True
    # 256 Tokens per KV Chunk
    export LMCACHE_CHUNK_SIZE=256
    # LMCache Server host
    export LMCACHE_REMOTE_URL="lm://localhost:65432"

    # How to serialize and deserialize KV cache on remote transmission
    export LMCACHE_REMOTE_SERDE="naive" # "naive" (default) or "cachegen"

**2. Configuration File**:

Passed in through ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml``

``LMCACHE_USE_EXPERIMENTAL`` MUST be set by environment variable directly.

Example ``config.yaml``:

.. code-block:: yaml

    # 256 Tokens per KV Chunk
    chunk_size: 256
    # LMCache Server host
    remote_url: "lm://localhost:65432"

    # How to serialize and deserialize KV cache on remote transmission
    remote_serde: "naive" # "naive" (default) or "cachegen"

Remote Storage Explanation:
----------------------------

LMCache's backend obeys the natural memory hierarchy of prioritizing CPU RAM offloading, then Local Storage
offloading, and finally remote offloading.

For LMCache to know how to create a connector to a remote backend, you must specify in
``remote_url`` a connector type followed by one or most host:port pairs (depending on what connector type is used).
If ``remote_url`` is set to ``None``, LMCache will not use any remote storage.

Example of LMCache Server ``remote_url``:

.. code-block:: yaml

    remote_url: "lm://localhost:65432"

Starting an LMCache Server:
---------------------------

The ``lmcache_server`` CLI entrypoint starts a remote LMCache server and comes with
the ``lmcache`` package.

.. code-block:: bash

    lmcache_server --host <host> --port <port> --device <device> --capacity <capacity>

    lmcache_server --host localhost --port 65432 --capacity 100

Currently, the only supported device is "cpu" (which is the default, so you don't need to specify it).

Remote Storage Example
-----------------------

.. _lmserver-prerequisites:

**Prerequisites:**

- A Machine with at least one GPU. You can adjust the max model length of your vllm instance depending on your GPU memory.

- vllm and lmcache installed (:doc:`Installation Guide <../../getting_started/installation>`)

- Hugging Face access to ``meta-llama/Meta-Llama-3.1-8B-Instruct``

.. code-block:: bash

    export HF_TOKEN=your_hugging_face_token

**Step 0. Set up a directory for this example:**

.. code-block:: bash

    mkdir lmcache-server-offload-example
    cd lmcache-server-offload-example

**Step 1. Start an LMCache Server:**

.. code-block:: bash

    lmcache_server --host localhost --port 65432 --capacity 100

**Step 2. Start a vLLM server with LMCache Server offloading enabled:**

Create an lmcache configuration file called: ``lmcache-server-offload.yaml``

.. code-block:: yaml

    chunk_size: 256
    remote_url: "lm://localhost:65432"
    remote_serde: "naive"

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_REMOTE_URL="lm://localhost:65432" \
    # LMCACHE_REMOTE_SERDE="naive"
    LMCACHE_CONFIG_FILE="lmcache-server-offload.yaml" \
    LMCACHE_USE_EXPERIMENTAL=True \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 16384 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

**Step 3. Test the setup:**

You can test the LMCache Server setup by making requests to your vLLM instance and verifying
that KV cache is being stored and retrieved from the remote LMCache server.

**Step 4. Clean up:**

Stop the LMCache server by terminating the ``lmcache_server`` process. 