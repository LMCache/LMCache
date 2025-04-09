.. _dev_doc_LMCache_Engine_Examples:

LMCache Engine Examples
-----------------------

This page provides programming examples of using the LMCache Engine, demonstrating how to use the core functionality for storing and retrieving cache data.

Store and Retrieve Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The LMCache Engine provides three primary operations:

* ``store``: Caches key-value tensors from transformer attention layers
* ``retrieve``: Fetches cached KV values for token sequences
* ``lookup``: Checks if token sequences are cached without retrieving the actual KV cache

Using the Lookup Method
^^^^^^^^^^^^^^^^^^^^^^

The ``lookup`` method allows you to efficiently check if tokens are in the cache without retrieving the actual KV cache values:

.. code-block:: python

    # Check if tokens are in the cache
    engine = LMCacheEngine(config, metadata)
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Store tokens and their KV cache
    engine.store(tokens, kv_cache)

    # Check how many prefix tokens are cached
    prefix_length = engine.lookup(tokens)
    # prefix_length == 10 (all tokens are cached)

    # Check with a longer sequence
    extended_tokens = torch.cat([tokens, torch.tensor([11, 12, 13])])
    prefix_length = engine.lookup(extended_tokens)
    # prefix_length == 10 (only the first 10 tokens are cached)

    # Check with a shorter sequence
    shorter_tokens = tokens[:5]
    prefix_length = engine.lookup(shorter_tokens)
    # prefix_length == 5 (the entire shorter sequence is cached)

    # Check an entirely new sequence
    new_tokens = torch.tensor([100, 101, 102])
    prefix_length = engine.lookup(new_tokens)
    # prefix_length == 0 (no tokens are cached)

Use Cases of ``lookup``:

1. **Efficiency**: Much faster than ``retrieve`` since it only checks for existence without transferring data
2. **Prefix Detection**: Identifies exactly how much of a token sequence is cached
3. **Cache-Aware Inference**: Applications can use this to decide:
   - Whether to use cached KV values or compute from scratch
   - Which portion of a prompt needs fresh computation
   - When to prefetch missing chunks

Distributed Caching Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For applications that need distributed caching across multiple servers, you can configure the LMCacheEngine with appropriate distributed settings:

.. code-block:: python

    from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
    from lmcache.cache_engine import LMCacheEngine

    # Configure for distributed operation
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        remote_url="lm://cache-server:65432",  # Remote cache server
        enable_p2p=True,                       # Enable peer-to-peer lookup
        lookup_url="redis://lookup-server:6379", # Redis for distributed lookup
        distributed_url="localhost:65433"      # This node's distributed endpoint
    )

    # Create metadata for your model
    metadata = LMCacheEngineMetadata(
        "my_model", 
        world_size=1, 
        worker_id=0, 
        fmt="vllm", 
        kv_dtype=torch.bfloat16, 
        kv_shape=(32, 2, 256, 8, 128)
    )

    # Initialize engine with distributed capabilities
    engine = LMCacheEngine(config, metadata)

Remote Lookup Methods
^^^^^^^^^^^^^^^^^^^^

There are multiple ways to perform remote lookups in a distributed caching environment:

1. Using Direct Lookup
~~~~~~~~~~~~~~~~~~~~~~

The standard ``lookup`` method works transparently with remote servers when properly configured:

.. code-block:: python

    # Check if tokens exist in local or remote cache
    prefix_length = engine.lookup(tokens)

When the engine is configured with ``enable_p2p=True`` and proper lookup/distributed URLs, the lookup will check both local and remote caches.

2. Using the Redis Lookup Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For applications that need direct access to lookup capability, you can use the RedisLookupServer:

.. code-block:: python

    import redis
    from lmcache.utils import CacheEngineKey

    # Connect to the Redis lookup server
    r = redis.Redis(host="lookup-server", port=6379, decode_responses=True)

    # Create a key for the token chunk you want to check
    key = CacheEngineKey(
        fmt="vllm",
        model_name="my_model",
        world_size=1,
        worker_id=0,
        chunk_hash="hash_of_token_chunk" # You'd need to compute this using the same method as LMCache
    )

    # Check if the key exists in the distributed cache
    location = r.get(key.to_string())
    if location:
        # The key exists in the cache at the server identified by 'location'
        host, port = location.split(":")
        print(f"Tokens are cached on server {host}:{port}")
    else:
        # The key is not in any cache
        print("Tokens are not cached")

3. Using the LMCServerConnector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For client applications that want to check cache state without a full LMCacheEngine instance:

.. code-block:: python

    from lmcache.storage_backend.connector import CreateConnector
    from lmcache.utils import CacheEngineKey

    # Connect to the LM server
    connector = CreateConnector("lm://cache-server:65432")

    # Create a key for a token chunk
    key = CacheEngineKey(
        fmt="vllm",
        model_name="my_model",
        world_size=1,
        worker_id=0,
        chunk_hash="hash_of_token_chunk" # You'd need to compute this using the same method as LMCache
    )

    # Check if key exists
    exists = connector.exists(key)
    if exists:
        print("Token chunk is cached")
    else:
        print("Token chunk is not cached")
