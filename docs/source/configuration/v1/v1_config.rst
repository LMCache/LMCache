.. _v1_config:

Configuring LMCache v1
======================

In addition to the configurations in v0, LMCache v1 needs also newer ones.
For the overlapping v0 configuration, please refer to :ref:`v0_config`.

.. note::
      KV Blending is not supported in LMCache v1 yet and will be added in future releases.

LMCache v1 uses the SAME two ways to configure:
   * Using a YAML configuration file
   * Using environment variables

Using a YAML configuration file
-------------------------------

The following are the list of configurations parameters that can be set for LMCache.
Configurations are set in the format of a YAML file.

.. code-block:: yaml

      # whether to enable peer-to-peer sharing
      # default is False
      enable_p2p: bool  
      # the url of the lookup server
      lookup_url: Optional[str] 
      # the url of the distributed server
      distributed_url: Optional[str]
      # experimental features in LMCache
      use_experimental: bool

This configuration file can be named as ``lmcache_config.yaml`` and passed to the LMCache 
using the ``LMCACHE_CONFIG_FILE`` environment variable as follows:

.. code-block:: console

      $ LMCACHE_CONFIG_FILE=./lmcache_config.yaml vllm serve <args>

Using environment variables
-------------------------------

Using environment variables is another way to configure LMCache. In addition to the configurations in v0, 
LMCache v1 has the following additional configurations:

.. code-block:: bash

      # whether to enable peer-to-peer sharing
      # default is False
      LM_CACHE_ENABLE_P2P: bool

      # the url of the lookup server
      LM_CACHE_LOOKUP_URL: Optional[str]

      # the url of the distributed server
      LM_CACHE_DISTRIBUTED_URL: Optional[str]

      # experimental features in LMCache
      LM_CACHE_USE_EXPERIMENTAL: bool
      

To run LMCache with the environment variables, you can do the following:

.. code-block:: bash

      export LM_CACHE_ENABLE_P2P=True
      export LM_CACHE_LOOKUP_URL="http://localhost:8000"
      export LM_CACHE_DISTRIBUTED_URL="http://localhost:8001"
      export LM_CACHE_USE_EXPERIMENTAL=True

      vllm serve <args>

You can wrap these lines in a file ``run.sh`` and run it as follows:

.. code-block:: console

      $ chmod +x run.sh
      $ bash ./run.sh
