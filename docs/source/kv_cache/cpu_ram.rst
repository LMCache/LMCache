CPU RAM
=======

CPU RAM and Local Storage are the two ways of offloading KV cache onto non-GPU
memory of the same machine that is running inference for your model.

There are primarily two ways to configure LMCache:
1. Environment Variables
2. Configuration YAML file (passed in through `LMCACHE_CONFIG_FILE=your-lmcache-config.yaml`)

Examples of how to configure CPU RAM offloading both ways:

1. Environment variables for LMCache:

`LMCACHE_USE_EXPERIMENTAL` MUST be set by environment variable directly.

.. code-block:: bash
    # Specify LMCache V1 (MUST be set by environment variable directly)
    export LMCACHE_USE_EXPERIMENTAL=True
    # 256 Tokens per KV Chunk
    export LMCACHE_CHUNK_SIZE=256
    # Enable CPU memory backend
    export LMCACHE_LOCAL_CPU=True
    # 5GB of Pinned CPU memory
    export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0

2. Configuration file for LMCache (e.g. `my-lmcache-config.yaml`):

.. code-block:: yaml
    # 256 Tokens per KV Chunk
    chunk_size: 256
    # Enable CPU memory backend
    local_cpu: true
    # 5GB of Pinned CPU memory
    max_local_cpu_size: 5.0

The`LMCACHE_MAX_LOCAL_CPU_SIZE` is the amount of page-locked (for fast GPU transfer)
CPU memory that LMCache will reserve and must be set to a number greater than 0 since the
local and remote backends use CPU RAM as an intermediate buffer when transferring
stored KV caches to the GPU.

It is recommended to *always* set `LMCACHE_USE_LOCAL_CPU=True` since this allows
all currently unused pinned CPU RAM that LMCache has reserved to be used for holding KV caches. If the pinned
CPU RAM is required for other operations (like disk or remote transfers), the CPU KV
caches will be evicted from the CPU to make space. The current eviction policy is LRU.

When `LMCACHE_USE_LOCAL_CPU=True` is used in conjunction with the disk backend or
a remote backend (see :doc:`Redis <./redis>`, :doc:`Mooncake <./mooncake>`, :doc:`Valkey <./valkey>`,
or :doc:`Infinistore <./infinistore>`), we can think of the CPU RAM as a "hot cache" that
will contain the "hottest" subset of KV caches that have most recently been
accessed in the Disk and Remote storage. Thus, the cache engine has a **prefetch** mechanism
to preload the KV caches for specified tokens into the pinned CPU RAM from the disk or
remote storage (*if* the KV caches for these tokens are already stored there). This helps
preemptively avoid the latency of the disk and remote KV storage if we predict these tokens
will be requested soon.

# Online Inference Example:

Let's feel the TTFT (time to first token) differential!

0. Prerequisites:
- A Machine with at least one GPU
- vllm and lmcache installed (:doc:`Installation Guide <../getting_started/installation>`)
- Hugging Face access to model ``meta-llama/Meta-Llama-3.1-8B-Instruct``
.. code-block:: bash

    export HF_TOKEN=your_hugging_face_token


Set up a directory for this example:

.. code-block:: bash

    mkdir lmcache-cpu-ram-example
    cd lmcache-cpu-ram-example

1. Prepare a long context! (long enough so that vllm's built-in prefix caching will
not be able to hold the KV cache in GPU memory and we need LMCache to help keep
it in CPU memory in this example):

.. code-block:: bash

    # 170000 bytes, 1695 words
    man bash | head -c 170000 > man-bash.txt

2. Start up a vLLM server with CPU offloading enabled:

`cpu-offload.yaml`

.. code-block:: yaml

    use_experimental: true
    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5.0

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the `LMCACHE_CONFIG_FILE` below:

.. code-block::bash

    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_LOCAL_CPU=True \
    # LMCACHE_MAX_LOCAL_CPU_SIZE=5.0 \
    LMCACHE_CONFIG_FILE="cpu-offload.yaml" \
    LMCACHE_USE_EXPERIMENTAL=True \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 8192 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

- ``--kv-transfer-config``: This is the parameter that actually tells vLLM to use LMCache for KV cache offloading.
    - ``kv_connector``: Specifies the LMCache connector for vLLM V1
    - ``kv_role``: Set to "kv_both" for both storing and loading KV cache (important because we will run two queries and the first will produce/store a KV cache while the second will consume/load that KV cache)

Once the open ai compatible server is running on default vllm port 8000, let's query it twice with the same long context!

Create a file called `query-twice.py` and paste the following code:

.. code-block:: python

    import time

    start_time = time.time()




Run ``python query-twice.py`` and you should see the following output:

.. code-block:: text

    TTFT with KV cache offloading: 1.234 seconds
    TTFT without KV cache offloading: 2.345 seconds
