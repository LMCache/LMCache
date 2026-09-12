Request Transport
=================

LMCache MP clients select the request transport from the LMCache server URL.
This transport carries control and cache-operation requests; it is separate
from ``--supported-transfer-mode``, which controls how KV data moves between
the engine worker and the LMCache server.

Transport schemes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Endpoint
     - Transport
     - Status
   * - ``host:port`` or ``tcp://host:port``
     - ZMQ over TCP
     - Supported. A URL without a scheme defaults to ``tcp://``.
   * - ``ipc://path`` or ``inproc://name``
     - ZMQ IPC or in-process
     - Recognized by the client factory; the standard LMCache server currently
       binds ZMQ over TCP.
   * - ``grpc://host:port`` or ``grpc+unix:///path``
     - gRPC
     - Not supported yet. gRPC support is planned soon.

gRPC schema development
-----------------------

The protobuf schemas for the planned gRPC transport live under
``lmcache/v1/multiprocess/transport/grpc_impl/protos``. Landing these schemas
and the manual generator does not add gRPC runtime dependencies, generate
bindings during package builds, or enable the gRPC client or server. ``grpc://``
endpoints remain unavailable until the runtime implementation lands.

For local schema development, the manual generator remains available under the
``_proto_gen`` package, but default installs, tests, and package builds do not
invoke it.

For a single vLLM connector, set the scheme in ``lmcache.mp.host`` and keep the
port in ``lmcache.mp.port``. The current ZMQ configuration is:

.. code-block:: json

   {
     "lmcache.mp.host": "tcp://localhost",
     "lmcache.mp.port": 5555
   }

When gRPC becomes available, selecting it will use the same configuration
shape with a ``grpc://`` host:

.. code-block:: json

   {
     "lmcache.mp.host": "grpc://localhost",
     "lmcache.mp.port": 5555
   }

For multiple servers, specify the scheme on every entry in
``lmcache.mp.server_urls``, for example
``tcp://host1:5555,tcp://host2:5555``. All clients and servers in a deployment
must use matching transports.
