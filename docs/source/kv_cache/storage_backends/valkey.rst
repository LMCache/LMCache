Valkey
======

Overview
--------

Valkey is an open source (BSD) high-performance key/value datastore and is a supported option for remote KV Cache offloading in LMCache.
Some other remote backends are :doc:`Mooncake <./mooncake>`, :doc:`Redis <./redis>`, and :doc:`InfiniStore <./infinistore>`.

Prerequisites
-------------

To use this connector, you need valkey-glide 2.0 or higher. Valkey Connector currently uses pipelining, which generally results in better RTT compared to Redis Connector.
Pipelining will also be implemented to the Redis Connector in the future.

.. code-block:: shell

    # Install Valkey-GLIDE (Minimum 2.0.0 or higher)
    $ pip install valkey-glide

Example Configurations
----------------------

Standalone-mode
~~~~~~~~~~~~~~~~

Basic Valkey Configuration (Standalone-mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_username: "Your username"
     valkey_password: "Your password"

Standalone-mode Valkey Configuration with database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_username: "Your username"
     valkey_password: "Your password"
     valkey_database: 0

Cluster-mode
~~~~~~~~~~~~~~~~

Cluster-mode Valkey Configuration (Endpoint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For example, the configuration endpoint in ElastiCache is as follows:

<cache-name>.<identifier>.clustercfg.<region>.cache.amazonaws.com

You need to add this DNS name in the <your host>.


.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_mode: "cluster"
     valkey_username: "Your username"
     valkey_password: "Your password"


Cluster-mode Valkey Configuration (Nodes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nodes are deployed directly and configured in cluster mode without connecting them via DNS names (CNAME).

In this scenario, you simply input multiple IP hosts and ports.

Example: 172.0.0.1:7001, 172.0.0.2:7002 ... 172.0.0.6:7006

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host 1>:<your port 1>, <your host 2>:<your port 2>, ... <your host N>:<your port N>"
   remote_serde: "naive"
   extra_config:
     valkey_mode: "cluster"
     valkey_username: "Your username"
     valkey_password: "Your password"

Cluster-mode Valkey Configuration with numbered databases (Valkey 9.0+ and Valkey-GLIDE 2.1+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Valkey connector supports numbered databases in both the Endpoint using DNS and the Nodes method using IP and port pairs.

.. code-block:: yaml

   chunk_size: 256
   remote_url: "valkey://<your host>:6379"
   remote_serde: "naive"
   extra_config:
     valkey_mode: "cluster"
     valkey_username: "Your username"
     valkey_password: "Your password"
     valkey_database: 1