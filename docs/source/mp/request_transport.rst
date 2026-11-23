Request Transport
=================

LMCache MP clients select the request transport from the LMCache server URL.
This transport carries control and cache-operation requests; it is separate
from ``--supported-transfer-mode``, which controls how KV data moves between
the engine worker and the LMCache server.

.. note::

   gRPC is the default request transport. ``lmcache server`` serves gRPC unless
   told otherwise, and a client URL without a scheme is read as ``grpc://``.
   ZMQ still works and still speaks the same protocol, but it is **deprecated**:
   the server and the client warn when it is selected, and the ZMQ client and
   server implementations will be removed in a future release. See
   :ref:`mp-migrating-from-zmq`.

Transport schemes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Endpoint
     - Transport
     - Status
   * - ``host:port``, ``grpc://host:port`` or ``grpc+unix:///path``
     - gRPC
     - **Default.** A URL without a scheme is read as gRPC.
   * - ``tcp://host:port``
     - ZMQ over TCP
     - Deprecated. Accepted during the transition; the server and the client
       warn when it is selected.
   * - ``ipc://path`` or ``inproc://name``
     - ZMQ IPC or in-process
     - Deprecated together with ZMQ. Recognized by the client factory, but the
       standard LMCache server binds TCP only, so these have no gRPC
       counterpart; use ``grpc://host:port``.

gRPC implementation and schema development
------------------------------------------

The gRPC client lives under ``lmcache/v1/client/transport/grpc_impl`` and the
gRPC servers under ``lmcache/v1/multiprocess/transport/grpc_impl``. Both take
endpoint parsing, the client-ID metadata key and the protobuf schemas from
``lmcache/v1/transport/grpc``. They expose the same named
``RequestClient`` operations and annotated engine-module handlers as ZMQ, while
protobuf defines the gRPC wire format. Schemas live in
``lmcache/v1/transport/grpc/protos``, and package builds generate their Python
bindings under ``lmcache/v1/transport/grpc/_proto_gen``.

After changing a schema, regenerate and validate all bindings with:

.. code-block:: bash

   pip install -r requirements/proto.txt
   python -m lmcache.v1.transport.grpc._proto_gen._generate

For a single vLLM connector, set the scheme in ``lmcache.mp.host`` and keep the
port in ``lmcache.mp.port``. A gRPC configuration is:

.. code-block:: json

   {
     "lmcache.mp.host": "grpc://localhost",
     "lmcache.mp.port": 5555
   }

The deprecated ZMQ transport uses the same configuration shape with a
``tcp://`` host:

.. code-block:: json

   {
     "lmcache.mp.host": "tcp://localhost",
     "lmcache.mp.port": 5555
   }

Start the server with the matching request transport:

.. code-block:: bash

   lmcache server --transport grpc --host localhost --port 5555
   lmcache server --transport zmq --host localhost --port 5555

Request API-key authentication
------------------------------

Request authentication is optional and enforced by both request transports.
Accepted server keys come from repeated ``--api-key-file``,
``LMCACHE_API_KEY_FILE``, and ``LMCACHE_API_KEY``; the sources are cumulative
for rotation. Clients send one explicit key or use one of the environment
sources. Authentication stays disabled when the server has no keys, preserving
existing deployments. The same key sources also protect the HTTP API.

ZMQ clients send the credential in a dedicated, versioned frame that the
server removes and verifies before decoding the RPC operation or payload
frames; streaming acknowledgements and cancellations carry the same envelope.
Enabling authentication requires clients that understand the versioned frame:
old servers cannot read the new credential frame, and clients that predate it
cannot talk to an authenticating server. Upgrade the deployment before
enabling keys.

On gRPC, a server interceptor checks Bearer metadata before protobuf request
deserialization and handler dispatch, and a client interceptor injects it for
unary and server-streaming calls. Missing or incorrect credentials return
``UNAUTHENTICATED``. The ``sync``, ``aio``, and ``native`` gRPC server modes all
enforce this policy.

.. code-block:: bash

   # Server: repeat --api-key-file to overlap old and new keys.
   lmcache server --transport zmq --host 0.0.0.0 --port 5555 \
       --api-key-file /run/secrets/lmcache-request-key

   lmcache server --transport grpc --host 0.0.0.0 --port 5555 \
       --api-key-file /run/secrets/lmcache-request-key

   # Client process, including vLLM or another engine integration.
   export LMCACHE_API_KEY_FILE=/run/secrets/lmcache-request-key

The environment can alternatively carry the value directly in
``LMCACHE_API_KEY``.

API keys do not encrypt request traffic: ZMQ sends the token in clear and the
default gRPC channel is insecure, leaving both the token and payload
unencrypted. Use a trusted network or an encrypted tunnel when the request
transport crosses an untrusted network. CURVE can be added as a future ZMQ
transport-security adapter, and mutual TLS can add encryption and
certificate-based peer identity to gRPC without replacing the API-key verifier.

For multiple servers, specify the scheme on every entry in
``lmcache.mp.server_urls``, for example
``grpc://host1:5555,grpc://host2:5555`` or
``tcp://host1:5555,tcp://host2:5555``. All clients and servers in a
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

.. _mp-migrating-from-zmq:

Migrating from ZMQ
------------------

A ``--transport zmq`` server keeps serving ZMQ, and a ``tcp://`` client URL
keeps reaching it. The default is what changes, so a deployment that relied on
the default on one side and named the transport on the other ends up on two
different transports.

.. warning::

   Clients from LMCache 0.5.7 and earlier cannot tell that a server speaks gRPC.
   Pointed at a server that now defaults to gRPC, they wait for the full
   ``lmcache.mp.mq_timeout`` (300 seconds by default) and then report that the
   server is not running. When the server and the engines are upgraded
   separately, upgrade the engines first, or start the upgraded server with
   ``--transport zmq`` until they catch up. Clients from 0.5.8 on fail at the
   first request with a message that names the transport.

Migrate in this order:

1. **Pick the transport the server should serve** and name it instead of relying
   on the default:

   .. code-block:: bash

      lmcache server --transport grpc --host 0.0.0.0 --port 5555

   The same applies to every other path that starts a server: the vLLM
   autostart derives the transport from the connector URL, and the operator and
   the test harnesses name theirs explicitly. With the operator, restart the
   vLLM pods after upgrading it, since they keep the connector config they
   started with.

2. **Point clients at that transport.** Set the scheme in the connector URL, or
   rely on the scheme-less default, which is now gRPC:

   .. code-block:: json

      {"lmcache.mp.host": "grpc://lmcache-server", "lmcache.mp.port": 5555}

   ``bench server --rpc-url`` and the SGLang ``mp_host`` setting take the same
   scheme. A ``tcp://`` URL keeps working until the removal release.

3. **Watch the logs.** Selecting ZMQ logs a deprecation warning on the server
   and in each client process. A deployment that logs no warning is already on
   gRPC.

4. **Remove the ZMQ settings** before the removal release. After it, a
   ``tcp://``, ``ipc://`` or ``inproc://`` request URL and a
   ``--transport zmq`` server fail with an error that points at this section
   rather than at a stack trace.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Release
     - Behaviour
   * - The release with the switch
     - gRPC is the default; ZMQ works and warns. Mixed fleets are fine as long
       as each client and its server agree.
   * - A later release
     - ZMQ client and server request transports are removed. HTTP APIs, the
       coordinator, KV transfer modes and the NIXL/Mooncake handshakes are
       unaffected -- those still use ``pyzmq``.
