CPU RAM
=======

CPU RAM and Local Storage are the two ways of offloading KV cache onto non-GPU
memory of the same machine that is running inference for your model.

There are primarily two ways to configure LMCache:
1. Environment Variables
2. Configuration YAML file (passed in through ``LMCACHE_CONFIG_FILE=your-lmcache-config.yaml``)

Examples of how to configure CPU RAM offloading both ways:

1. Environment variables for LMCache:

``LMCACHE_USE_EXPERIMENTAL`` MUST be set by environment variable directly.

.. code-block:: bash
    # Specify LMCache V1 (MUST be set by environment variable directly)
    export LMCACHE_USE_EXPERIMENTAL=True
    # 256 Tokens per KV Chunk
    export LMCACHE_CHUNK_SIZE=256
    # Enable CPU memory backend
    export LMCACHE_LOCAL_CPU=True
    # 5GB of Pinned CPU memory
    export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0

2. Configuration file for LMCache (e.g. ``my-lmcache-config.yaml``):

``LMCACHE_USE_EXPERIMENTAL`` MUST be set by environment variable directly.

.. code-block:: yaml
    # 256 Tokens per KV Chunk
    chunk_size: 256
    # Enable CPU memory backend
    local_cpu: true
    # 5GB of Pinned CPU memory
    max_local_cpu_size: 5.0

The ``LMCACHE_MAX_LOCAL_CPU_SIZE`` is the amount of page-locked (for fast GPU transfer)
CPU memory that LMCache will reserve and must be set to a number greater than 0 since the
local and remote backends use CPU RAM as an intermediate buffer when transferring
stored KV caches to the GPU.

It is recommended to *always* set ``LMCACHE_USE_LOCAL_CPU=True`` since this allows
all currently unused pinned CPU RAM that LMCache has reserved to be used for holding KV caches. If the pinned
CPU RAM is required for other operations (like disk or remote transfers), the CPU KV
caches will be evicted from the CPU to make space. The current eviction policy is LRU.

When ``LMCACHE_USE_LOCAL_CPU=True`` is used in conjunction with the disk backend or
a remote backend (see :doc:`Redis <./redis>`, :doc:`Mooncake <./mooncake>`, :doc:`Valkey <./valkey>`,
or :doc:`Infinistore <./infinistore>`), we can think of the CPU RAM as a "hot cache" that
will contain the "hottest" subset of KV caches that have most recently been
accessed in the Disk and Remote storage. Thus, the cache engine has a **prefetch** mechanism
to preload the KV caches for specified tokens into the pinned CPU RAM from the disk or
remote storage (*if* the KV caches for these tokens are already stored there). This helps
preemptively avoid the latency of the disk and remote KV storage if we predict these tokens
will be requested soon.

Online Inference Example
-----------------------

Let's feel the TTFT (time to first token) differential!

0. Prerequisites:
- A Machine with at least one GPU. Adjust the max model length of your vllm
instance depending on your GPU memory and the long context you want to use.
- vllm and lmcache installed (:doc:`Installation Guide <../getting_started/installation>`)
- Hugging Face access to model ``meta-llama/Meta-Llama-3.1-8B-Instruct``

.. code-block:: bash

    export HF_TOKEN=your_hugging_face_token


Set up a directory for this example:

.. code-block:: bash

    mkdir lmcache-cpu-ram-example
    cd lmcache-cpu-ram-example

1. Prepare a long context! We want it to be long enough
so that vllm's prefix caching will not be able to hold the KV caches in
GPU memory and we rely on LMCache to help keep it in CPU memory:

Example:

.. code-block:: bash

    man bash > man-bash.txt

2. Start up a vLLM server with CPU offloading enabled:

``cpu-offload.yaml``

.. code-block:: yaml

    chunk_size: 256
    local_cpu: true
    max_local_cpu_size: 5.0

If you don't want to use a config file, uncomment the first three environment variables
and then comment out the ``LMCACHE_CONFIG_FILE`` below:

.. code-block:: bash

    # LMCACHE_CHUNK_SIZE=256 \
    # LMCACHE_LOCAL_CPU=True \
    # LMCACHE_MAX_LOCAL_CPU_SIZE=5.0 \
    LMCACHE_CONFIG_FILE="cpu-offload.yaml" \
    LMCACHE_USE_EXPERIMENTAL=True \
    vllm serve \
        meta-llama/Llama-3.1-8B-Instruct \
        --max-model-len 16384 \
        --kv-transfer-config \
        '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

- ``--kv-transfer-config``: This is the parameter that actually tells vLLM to use LMCache for KV cache offloading.
    - ``kv_connector``: Specifies the LMCache connector for vLLM V1
    - ``kv_role``: Set to "kv_both" for both storing and loading KV cache (important because we will run two queries and the first will produce/store a KV cache while the second will consume/load that KV cache)

Once the open ai compatible server is running on default vllm port 8000, let's query it twice with the same long context!

Create a file called ``query-twice.py`` and paste the following code:

.. code-block:: python

    import time
    from openai import OpenAI
    from transformers import AutoTokenizer

    client = OpenAI(
        api_key="dummy-key",  # required by OpenAI client even for local servers
        base_url="http://localhost:8000/v1"
    )

    models = client.models.list()
    model = models.data[0].id

    # 119512 characters total
    # 26054 tokens total
    long_context = ""
    with open("man-bash.txt", "r") as f:
        long_context = f.read()

    # a truncation of the long context for the --max-model-len 16384
    # if you increase the --max-model-len, you can decrease the truncation i.e.
    # use more of the long context
    long_context = long_context[:70000]

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    question = "Summarize bash in 2 sentences."

    prompt = f"{long_context}\n\n{question}"

    print(f"Number of tokens in prompt: {len(tokenizer.encode(prompt))}")

    def query_and_measure_ttft():
        start = time.perf_counter()
        ttft = None
        server_message = []

        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=0.7,
            stream=True,
        )

        for chunk in chat_completion:
            chunk_message = chunk.choices[0].delta.content
            if chunk_message is not None:
                if ttft is None:
                    ttft = time.perf_counter()
                print(chunk_message, end="", flush=True)
                server_message.append(chunk_message)

        print("\n")  # New line after streaming
        return ttft - start, "".join(server_message)

    print("Querying vLLM server with cold LMCache")
    cold_ttft, cold_response = query_and_measure_ttft()
    print(f"Cold TTFT: {cold_ttft:.3f} seconds")

    print("\nQuerying vLLM server with warm LMCache")
    warm_ttft, warm_response = query_and_measure_ttft()
    print(f"Warm TTFT: {warm_ttft:.3f} seconds")

    print(f"\nTTFT Improvement: {(cold_ttft - warm_ttft):.3f} seconds ({(cold_ttft/warm_ttft):.1f}x faster)")

.. code-block:: bash
    python query-twice.py

Since we're in streaming mode, you'll be able to feel the TTFT differential in
real time!

Example Output:

.. code-block:: text

    Number of tokens in prompt: 15376
    Querying vLLM server with cold LMCache
    Bash is a command-line interpreter that executes commands read from the
    standard input or from a file, and it incorporates features from the Korn
    and C shells. It is an sh-compatible command language interpreter that
    can be configured to be POSIX-conformant by default, and it provides a
    wide range of features, including shell functions, arrays, and conditional
    expressions, as well as built-in commands and options for customizing
    its behavior.

    Cold TTFT: 5.632 seconds

    Querying vLLM server with warm LMCache
    Bash is a Unix shell and command-line interpreter that reads and
    executes commands from the standard input or a file, incorporating features
    from the Korn and C shells. It is designed to be a conformant implementation
    of the IEEE POSIX specification and can be configured to be POSIX-conformant
    by default.

    Warm TTFT: 0.144 seconds

    TTFT Improvement: 5.487 seconds (39.0x faster)

**Tips:**
- If you want to run the ``query-twice.py`` script multiple times, you'll need to
either restart the vLLM LMCache server or change the prefix of the context you pass in
since you've already warmed LMCache.
- The max model length here was decided by running an L4 with only 23GB of GPU
memory. If you have more memory, you can increase the max model length and modify
``query-twice.py`` to use more of the long context. LMCache TTFT improvement becomes
more pronounced as the context length increases!