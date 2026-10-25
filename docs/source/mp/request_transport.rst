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
     - Supported.

gRPC implementation and schema development
------------------------------------------

The gRPC client and server live under
``lmcache/v1/multiprocess/transport/grpc_impl``. They expose the same named
``RequestClient`` operations and annotated engine-module handlers as ZMQ, while
protobuf defines the gRPC wire format. Schemas live in the sibling ``protos``
directory, and package builds generate their Python bindings under
``_proto_gen``.

After changing a schema, regenerate and validate all bindings with:

.. code-block:: bash

   pip install -r requirements/proto.txt
   python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate

For a single vLLM connector, set the scheme in ``lmcache.mp.host`` and keep the
port in ``lmcache.mp.port``. A ZMQ configuration is:

.. code-block:: json

   {
     "lmcache.mp.host": "tcp://localhost",
     "lmcache.mp.port": 5555
   }

To select gRPC, use the same configuration shape with a ``grpc://`` host:

.. code-block:: json

   {
     "lmcache.mp.host": "grpc://localhost",
     "lmcache.mp.port": 5555
   }

Start the server with the matching request transport:

.. code-block:: bash

   lmcache server --transport zmq --host localhost --port 5555
   lmcache server --transport grpc --host localhost --port 5555

gRPC API-key authentication
---------------------------

gRPC authentication is optional. Accepted server keys come from repeated
``--api-key-file``, ``LMCACHE_API_KEY_FILE``, and
``LMCACHE_API_KEY``. Sources are cumulative for rotation. Clients send
one explicit key or use one of the environment sources. A server interceptor checks
Bearer metadata before protobuf request deserialization and handler dispatch;
a client interceptor injects it for unary and server-streaming calls. Missing
or incorrect credentials return gRPC ``UNAUTHENTICATED``. Authentication stays
disabled when the server has no keys. Both ``--grpc-server-mode sync`` and
``--grpc-server-mode aio`` enforce this authentication policy.

The same key sources also protect the HTTP API. With the default ZMQ
transport, the server continues to start and warns that only HTTP is protected;
ZMQ requests remain unauthenticated until ZMQ authentication is implemented.
ZMQ clients ignore the shared key environment variables so existing engines
can inherit an HTTP-protected server's environment. Explicitly passing
``api_key`` to a ZMQ client is rejected rather than silently ignored.

.. code-block:: bash

   lmcache server --transport grpc --host 0.0.0.0 --port 5555 \
       --api-key-file /run/secrets/lmcache-request-key

   export LMCACHE_API_KEY_FILE=/run/secrets/lmcache-request-key

The default gRPC channel is insecure, so the token and payload are not
encrypted. Use TLS when traffic crosses an untrusted network. Mutual TLS can
later add transport encryption and certificate-based peer identity without
replacing the API-key verifier.

For multiple servers, specify the scheme on every entry in
``lmcache.mp.server_urls``, for example
``tcp://host1:5555,tcp://host2:5555`` or
``grpc://host1:5555,grpc://host2:5555``. All clients and servers in a
deployment must use matching transports.


IPC descriptor compatibility
----------------------------

Device IPC wrappers use versioned, data-only MessagePack descriptors in both
transports. Legacy pickle descriptors are rejected; upgrade workers and servers
together. MUSA handles must use MessagePack scalars, lists and string-keyed maps;
opaque SDK objects require an explicit codec and are rejected.

This closes the device-wrapper deserialization path in CVE-2026-105192. The
engine-driven tensor-data path uses a separate typed inline codec that shares
the descriptor dtype allowlist. API-key authentication is optional and does
not replace transport encryption; keep unauthenticated or insecure endpoints
restricted to trusted networks.
