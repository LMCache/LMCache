Server Module Plugins
=====================

The multiprocess ``lmcache server`` is composed from transport-neutral
server modules. Built-in modules handle lookup, management, transfer, P2P,
and optional experimental flows. A server module plugin lets an installed
Python package add extra modules at startup without modifying LMCache source.

This mechanism is explicit opt-in. Installing a package is not enough to
change server behavior; the operator must pass ``--server-module``.

When to use it
--------------

Use a server module plugin when implementation should live in another package.
Examples include vendor-specific implementations, experimental handlers, or
site-local observability/control modules that reuse LMCache's shared
``MPCacheServerContext``.

For new extension protocols, prefer a transport-specific service when the
extension owns a real client protocol. A plugin can register its own generated
gRPC service or its own ZMQ service during LMCache server startup, while the
same package also returns server modules for status, cleanup, and shared
``MPCacheServerContext`` access.

For small control-plane calls that do not need a first-class service, plugins
may use the stable ``SERVER_MODULE_CALL`` envelope. The core wire contract
stays fixed across transports:

- ``method`` is a namespaced string such as ``"my_package.echo"``.
- ``payload`` is plugin-owned request bytes.
- the response contains ``success``, plugin-owned response ``payload`` bytes,
  and a human-readable ``error`` string.

This gives out-of-tree packages a lightweight fallback namespace without
appending a new ``RequestType`` enum value or generated gRPC method for every
plugin feature. If a feature needs a high-throughput or strongly typed API,
register a package-owned gRPC/ZMQ service. If every LMCache client should know
about the API, add that request type, payload, response, and gRPC schema to
LMCache core instead.

Configuration
-------------

Pass one JSON object per ``--server-module`` flag:

.. code-block:: bash

   lmcache server \
       --server-module '{
         "module_path": "my_package.server_module",
         "factory_name": "build_server_modules",
         "config": {
           "mode": "shadow"
         }
       }'

``factory_name`` defaults to ``build_server_modules`` and ``config`` defaults
to an empty object. The flag can be repeated, and a single flag may contain a
JSON list of module specs.

.. list-table:: Server module spec fields
   :header-rows: 1
   :widths: 22 16 62

   * - Field
     - Required
     - Description
   * - ``module_path``
     - yes
     - Dotted Python import path containing the module factory.
   * - ``factory_name``
     - no
     - Callable name inside ``module_path``. Defaults to
       ``build_server_modules``.
   * - ``config``
     - no
     - Plugin-specific JSON object passed to the factory.

Factory contract
----------------

The factory receives one ``ServerModuleBuildContext`` and returns
``ServerModuleComponents``. For backwards compatibility, returning one module,
a sequence of modules, or ``None`` is also accepted:

.. code-block:: python

   from lmcache.v1.multiprocess.server_module import (
       ServerModuleBuildContext,
       ServerModuleComponents,
   )


   def build_server_modules(ctx: ServerModuleBuildContext) -> ServerModuleComponents:
       module = MyServerModule(ctx.server_context, ctx.config)
       return ServerModuleComponents(
           modules=[module],
           grpc_service_registrars=[module.register_grpc_services],
           zmq_service_registrars=[module.register_zmq_services],
       )

A service-only package can return no modules:

.. code-block:: python

   def build_server_modules(ctx: ServerModuleBuildContext) -> ServerModuleComponents:
       return ServerModuleComponents(
           grpc_service_registrars=[build_echo_grpc_service(ctx)],
       )

``ServerModuleBuildContext`` contains:

- ``server_context``: the shared ``MPCacheServerContext``.
- ``mp_config``: parsed ``MPServerConfig``.
- ``coordinator_config``: parsed coordinator configuration.
- ``modules``: built-in modules, plus modules returned by earlier plugin
  factories in command-line order.
- ``config``: the JSON object from this plugin's server-module spec.

Module contract
---------------

A returned module must expose the regular server module contract:

.. code-block:: python

   @property
   def context(self): ...

   def report_status(self) -> dict: ...

   def close(self) -> None: ...

Request handlers are ordinary transport-neutral handlers decorated with
``@request_handler``:

.. code-block:: python

   from lmcache.v1.multiprocess.request_handler import request_handler
   from lmcache.v1.multiprocess.protocol import RequestType
   from lmcache.v1.multiprocess.protocols.base import HandlerType


   class MyServerModule:
       def __init__(self, ctx, config):
           self._ctx = ctx
           self._config = config

       @property
       def context(self):
           return self._ctx

       def report_status(self) -> dict:
           return {"my_module": {"is_healthy": True}}

       def close(self) -> None:
           return None

Plugin modules can also expose extension protocol methods with
``@server_module_handler``:

.. code-block:: python

   import msgspec

   from lmcache.v1.multiprocess.server_module import server_module_handler


   class EchoRequest(msgspec.Struct):
       text: str


   class EchoResponse(msgspec.Struct):
       text: str


   class MyServerModule:
       # context/report_status/close omitted for brevity

       @server_module_handler("my_package.echo")
       def echo(self, payload: bytes) -> bytes:
           request = msgspec.msgpack.decode(payload, type=EchoRequest)
           return msgspec.msgpack.encode(EchoResponse(text=request.text))

Plugin modules are appended after built-in modules. If two modules register
the same existing request type, the later module wins for transport handler
registration. If two plugin modules register the same ``SERVER_MODULE_CALL``
method name, startup fails.

Transport-specific services
---------------------------

When an extension owns its protocol, return transport registrars in
``ServerModuleComponents``. LMCache calls the registrar that matches the
configured transport before the request server starts.

For gRPC, register generated services against the concrete ``grpc.Server``:

.. code-block:: python

   from my_package.protos import echo_pb2_grpc


   def register_echo_grpc_service(server) -> None:
       echo_pb2_grpc.add_EchoServiceServicer_to_server(
           MyEchoServicer(),
           server,
       )

For ZMQ, register or start a package-owned ZMQ service against the concrete
``MessageQueueServer``:

.. code-block:: python

   def register_echo_zmq_service(server) -> None:
       echo_service = MyZmqEchoService(
           context=server.ctx,
           bind_url="tcp://127.0.0.1:6001",
       )
       echo_service.start()

If a ZMQ registrar starts a sidecar socket, the plugin should also return a
module whose ``close()`` method stops that sidecar. Do not use
``register_zmq_services`` to add ``HandlerType.BLOCKING`` handlers to the core
``MessageQueueServer`` unless the registrar also assigns an executor with
``add_normal_thread_pool`` or ``add_affinity_thread_pool`` before startup.

For compatibility with simple module-shaped plugins, LMCache also calls
``register_grpc_services(server)`` and ``register_zmq_services(server)``
methods when a returned module exposes them. New packages should prefer the
explicit ``ServerModuleComponents`` fields so modules and transport services
stay separate.

Envelope client calls
---------------------

Both ZMQ and gRPC clients expose ``server_module_call``:

.. code-block:: python

   import msgspec

   from lmcache.v1.multiprocess.protocols.server_module import (
       ServerModuleCallRequest,
   )


   payload = msgspec.msgpack.encode(EchoRequest(text="hello"))
   response = client.server_module_call(
       ServerModuleCallRequest(method="my_package.echo", payload=payload)
   ).result(timeout=5)
   if not response.success:
       raise RuntimeError(response.error)
   echo = msgspec.msgpack.decode(response.payload, type=EchoResponse)

Liveness and cleanup
--------------------

If a plugin owns per-worker state that should be pinged and reaped by
``ManagementModule``, implement the ``InstanceLivenessTarget`` method set:

- ``touch_instance(instance_id)``
- ``reap_stale_instances(reap_timeout_s, registration_grace_s)``
- ``tracked_instance_count()``
- ``drop_instance_state(instance_id)``

LMCache detects that method set when loading plugin modules and includes the
module in the management reaper targets.
