.. _config:

Configuring LMCache
====================

There are two possible ways to configure LMCache:
   * Using a YAML configuration file
   * Using environment variables

Using a YAML configuration file
-------------------------------

The following are the list of configurations parameters that can be set for LMCache.
Configurations are set in the format of a YAML file.

.. code-block:: yaml

      # The size of the chunk as an integer 
      # (set to 256 by default)
      chunk_size: int

      # The local KV cache device to use (set to "cuda" by default)
      # Possible values: "cpu", "cuda", "file://local_disk/"
      local_device: Optional[str]

      # The maximum size of the local KV cache as an integer (GB)
      # Set to 5 by default
      max_local_cache_size: int

      # Remote URL for the storage backend (can be redis or redis-sentinel)
      # Should have the format url://<host>:<port>
      # E.g. redis://localhost:65432
      # E.g. redis-sentinel://localhost:26379 
      remote_url: Optional[str]

      # The remote serde for the backend
      # Can be "cachegen", "torch", "safetensor", "fast"
      remote_serde: Optional[str]

      # Whether retrieve() is pipelined or not
      # Set to False by default
      pipelined_backend: bool

      # Whether to save the decode cache
      # Set to False by default
      save_decode_cache: bool 

      # Whether to enable KV cache blending
      # Set to False by default
      enable_blending: bool  

      # The recompute ratio if KV cache blending is enabled
      # Set to 0.5 by default 
      blend_recompute_ratio: float

      # The minimum number of tokens for blending
      # Set to 256 by default
      blend_min_tokens: int  

This configuration file can be named as ``lmcache_config.yaml`` and passed to the LMCache 
using the ``LMCACHE_CONFIG_FILE`` environment variable as follows:

.. code-block:: console

      $ LMCACHE_CONFIG_FILE=./lmcache_config.yaml lmcache_vllm serve <args>

Using environment variables
-------------------------------

The following are the list of environment variables that can be set for LMCache.

.. code-block:: bash

      # The size of the chunk as an integer 
      # (set to 256 by default)
      LM_CACHE_CHUNK_SIZE: int

      # The local KV cache device to use (set to "cuda" by default)
      # Possible values: "cpu", "cuda", "file://local_disk/"
      LM_CACHE_LOCAL_DEVICE: Optional[str]

      # The maximum size of the local KV cache as an integer (GB)
      # Set to 5 by default
      LM_CACHE_MAX_LOCAL_CACHE_SIZE: int

      # Remote URL for the storage backend (can be redis or redis-sentinel)
      # Should have the format url://<host>:<port>
      # E.g. redis://localhost:65432
      # E.g. redis-sentinel://localhost:26379 
      LM_CACHE_REMOTE_URL: Optional[str]

      # The remote serde for the backend
      # Can be "cachegen", "torch", "safetensor", "fast"
      LM_CACHE_REMOTE_SERDE: Optional[str]

      # Whether retrieve() is pipelined or not
      # Set to False by default
      LM_CACHE_PIPELINED_BACKEND: bool

      # Whether to save the decode cache
      # Set to False by default
      LM_CACHE_SAVE_DECODE_CACHE: bool 

      # Whether to enable KV cache blending
      # Set to False by default
      LM_CACHE_ENABLE_BLENDING: bool  

      # The recompute ratio if KV cache blending is enabled
      # Set to 0.5 by default 
      LM_CACHE_BLEND_RECOMPUTE_RATIO: float

      # The minimum number of tokens for blending
      # Set to 256 by default
      LM_CACHE_BLEND_MIN_TOKENS: int

To run LMCache with the environment variables, you can do the following:

.. code-block:: bash

      export LM_CACHE_CHUNK_SIZE=256
      export LM_CACHE_LOCAL_DEVICE="cuda"
      export LM_CACHE_MAX_LOCAL_CACHE_SIZE=5
      export LM_CACHE_REMOTE_URL="redis://localhost:65432"
      export LM_CACHE_REMOTE_SERDE="cachegen"
      export LM_CACHE_PIPELINED_BACKEND=False
      export LM_CACHE_SAVE_DECODE_CACHE=False
      export LM_CACHE_ENABLE_BLENDING=False
      export LM_CACHE_BLEND_RECOMPUTE_RATIO=0.5
      export LM_CACHE_BLEND_MIN_TOKENS=256

      lmcache_vllm serve <args>

You can wrap these lines in a file ``run.sh`` and run it as follows:

.. code-block:: console

      $ chmod +x run.sh
      $ bash ./run.sh
