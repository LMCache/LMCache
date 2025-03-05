.. _backend:

Selecting a backend
===================

LMCache supports multiple backends for storing the KV caches of reusable texts. The backends are 
configured using YAML files. The following backends are supported:

* Launch a Redis backend? Create a YAML file with the following configuration:

.. code-block:: yaml

   chunk_size: 256
   local_device: "cpu"
   remote_url: "redis://localhost:65432"
   remote_serde: "cachegen"

   # Whether retrieve() is pipelined or not
   pipelined_backend: False

* Launch a storage disk backend? Create a YAML file with the following configuration:

.. code-block:: yaml

   chunk_size: 256
   local_device: "file://local_disk/" # The path to the local disk

* Launch a local CPU backend? Create a YAML file with the following configuration:

.. code-block:: yaml

   chunk_size: 256
   local_device: "cpu"

   # Whether retrieve() is pipelined or not
   pipelined_backend: False

* Launch a local GPU backend? Create a YAML file with the following configuration:

.. code-block:: yaml

   chunk_size: 256
   local_device: "cuda"

.. note:: 

   In the configuration yaml files, the remote_url specifies the remote backend. Please remember to start remote server before starting vLLM.

   .. code-block:: console

      $ lmcache_server <HOST> <PORT>
      $ redis-server --bind <HOST> --port <PORT>

.. note::

   Different serializers and deserializers can be used for the backend's ``remote_serde``. 
   The default is ``cachegen``. Other options include ``torch``, ``safetensor``, and ``fast``.

.. note::

   Pipelined backend is used to pipeline the retrieve() calls. This can be useful when the backend is slow.
   Can be set to True or False.

Once the backend is configured, you can start the vLLM instance with the LMCache config file, similar to
as shown in :ref:`launching`.

.. code-block:: console

   $ LMCACHE_CONFIG_FILE=backend_type.yaml CUDA_VISIBLE_DEVICES=0 python offline_inference.py
