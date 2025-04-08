.. _v1_config:

Configuring LMCache v1
======================

In addition to the some configurations in v0, LMCache v1 needs also newer ones.
Please use this list of configurations to set up LMCache v1.
For V0, please refer to :ref:`v0_config`.

.. note::
      KV Blending is not supported in LMCache v1 yet and will be added in future releases.

LMCache v1 uses the SAME two ways to configure:
   * Using a YAML configuration file
   * Using environment variables

.. note::
      To turn on the experimental features in LMCache v1, you can set the environment variable
      ``LMCACHE_USE_EXPERIMENTAL=True``. This feature can only be set via environment variables.

Using a YAML configuration file
-------------------------------

The following are the list of configurations parameters that can be set for LMCache.
Configurations are set in the format of a YAML file.

.. code-block:: yaml

      # The size of the chunk as an integer 
      # (set to 256 by default)
      chunk_size: int

      # The local KV disk path
      local_disk: Optional[str]

      # The maximum size of the local KV cache as an integer (GB)
      # Set to 0 by default
      max_local_disk_size: int

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

      # whether to enable peer-to-peer sharing
      # default is False
      enable_p2p: bool  
      # the url of the lookup server
      lookup_url: Optional[str] 
      # the url of the distributed server
      distributed_url: Optional[str]

This configuration file can be named as ``lmcache_config.yaml`` and passed to the LMCache 
using the ``LMCACHE_CONFIG_FILE`` environment variable as follows:

.. code-block:: console

      $ LMCACHE_CONFIG_FILE=./lmcache_config.yaml vllm serve <args>

Using environment variables
-------------------------------

Using environment variables is another way to configure LMCache. In addition to the configurations in v0, 
LMCache v1 has the following additional configurations:

.. code-block:: bash

      # The size of the chunk as an integer 
      # (set to 256 by default)
      LMCACHE_CHUNK_SIZE: int

      LMCACHE_LOCAL_DISK: Optional[str]

      # The maximum size of the local KV cache as an integer (GB)
      # Set to 5 by default
      LMCACHE_MAX_LOCAL_DISK_SIZE: int

      # Remote URL for the storage backend (can be redis or redis-sentinel)
      # Should have the format url://<host>:<port>
      # E.g. redis://localhost:65432
      # E.g. redis-sentinel://localhost:26379 
      LMCACHE_REMOTE_URL: Optional[str]

      # The remote serde for the backend
      # Can be "cachegen", "torch", "safetensor", "fast"
      LMCACHE_REMOTE_SERDE: Optional[str]

      # Whether retrieve() is pipelined or not
      # Set to False by default
      LMCACHE_PIPELINED_BACKEND: bool

      # Whether to save the decode cache
      # Set to False by default
      LMCACHE_SAVE_DECODE_CACHE: bool 

      # whether to enable peer-to-peer sharing
      # default is False
      LMCACHE_ENABLE_P2P: bool

      # the url of the lookup server
      LMCACHE_LOOKUP_URL: Optional[str]

      # the url of the distributed server
      LMCACHE_DISTRIBUTED_URL: Optional[str]


To run LMCache with the environment variables, you can do the following:

.. code-block:: bash


      export LMCACHE_CHUNK_SIZE=256
      export LMCACHE_LOCAL_DEVICE="cuda"
      export LMCACHE_MAX_LOCAL_CACHE_SIZE=5
      export LMCACHE_REMOTE_URL="redis://localhost:65432"
      export LMCACHE_REMOTE_SERDE="cachegen"
      export LMCACHE_PIPELINED_BACKEND=False
      export LMCACHE_SAVE_DECODE_CACHE=False
      export LMCACHE_ENABLE_P2P=True
      export LMCACHE_LOOKUP_URL="http://localhost:8000"
      export LMCACHE_DISTRIBUTED_URL="http://localhost:8001"
      export LMCACHE_USE_EXPERIMENTAL=True

      vllm serve <args>

You can wrap these lines in a file ``run.sh`` and run it as follows:

.. code-block:: console

      $ chmod +x run.sh
      $ bash ./run.sh
