Server Module Plugins
=====================

The multiprocess ``lmcache server`` is composed from transport-neutral
server modules. Built-in modules handle lookup, management, transfer, P2P,
KV events, and blend. A server module plugin lets an installed Python package
add extra modules at startup without modifying LMCache source.

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
same module also provides status, cleanup, and shared
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
         "config": {
           "mode": "shadow"
         }
       }'

``config`` defaults to an empty object. The flag can be repeated, and a single
flag may contain a JSON list of module specs.

.. list-table:: Server module spec fields
   :header-rows: 1
   :widths: 22 16 62

   * - Field
     - Required
     - Description
   * - ``module_path``
     - yes
     - Dotted Python import path defining one or more concrete
       ``DiscoverableModule`` subclasses.
   * - ``config``
     - no
     - Plugin-specific JSON object available through
       ``ModuleBuildContext.config_for()``.

Opt-in modules shipped with LMCache
-----------------------------------

Some modules ship inside the LMCache package but are not built by default.
They live under ``lmcache.v1.multiprocess.plugin_modules``, outside the
built-in discovery root ``lmcache.v1.multiprocess.modules``, so the server
never composes them on its own. Load one exactly like an out-of-tree module,
by naming its Python module:

.. code-block:: bash

   lmcache server \
       --supported-transfer-mode auto \
       --server-module '{"module_path": "lmcache.v1.multiprocess.plugin_modules.qstore"}'

Currently shipped:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module path
     - Purpose
   * - ``lmcache.v1.multiprocess.plugin_modules.qstore``
     - ``QStoreModule``: stores per-layer query tensors captured by the vLLM
       connector (``"lmcache.mp.transfer_intermediate_tensors": true``).
       Requires the LMCache-driven transfer path; see :doc:`/mp/sdk`.

Module contract
---------------

External modules use the same discovery, dependency, construction, and runtime
contract as built-ins. LMCache imports ``module_path``, finds its concrete
``DiscoverableModule`` subclasses, merges them into the built-in dependency
graph, and calls ``create(build_ctx)``. There is no external-only base class or
factory contract: follow the existing ``DiscoverableModule`` implementations
under ``lmcache.v1.multiprocess.modules``. Plugin-specific configuration is
available from ``build_ctx.config_for(module_class)``.

Dependencies are class references, just as for built-ins. An external module
may depend on a built-in class or on another explicitly loaded external class.
Every dependency must be included in the discovered/configured class set.
At startup, LMCache logs every constructed module and its declared
dependencies as ``name(dependencies=[...])``.
Handlers are registered in construction order, so a dependent module can
override its dependency's handler. Modules close in reverse construction order;
``ManagementModule`` stops its reaper before liveness targets release their state.

Replacing a built-in module
---------------------------

A configured external module can replace a built-in module by subclassing it
and retaining the same ``module_name``. The built-in class is not constructed,
and dependencies declared with that built-in class resolve to the external
instance:

.. code-block:: python

   from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
       LMCacheDrivenTransferModule,
   )


   class MyTransferModule(LMCacheDrivenTransferModule):
       # Inherits module_name = "lmcache_driven_transfer".
       ...

Only explicitly configured external modules can replace built-ins. Startup
fails if an external module uses a built-in name without subclassing that
built-in class, or if multiple configured modules use the same name.

Plugin modules can also expose extension protocol methods with
``@server_module_handler``:

.. code-block:: python

   import msgspec

   from lmcache.v1.multiprocess.ext_server_module import server_module_handler


   class EchoRequest(msgspec.Struct):
       text: str


   class EchoResponse(msgspec.Struct):
       text: str


   class MyServerModule(DiscoverableModule):
       # context/report_status/close omitted for brevity

       @server_module_handler("my_package.echo")
       def echo(self, payload: bytes) -> bytes:
           request = msgspec.msgpack.decode(payload, type=EchoRequest)
           return msgspec.msgpack.encode(EchoResponse(text=request.text))

If two modules register the same server-module method name, of
either kind, startup fails.

Transport-specific services
---------------------------

When an extension owns its protocol, expose the registrar matching the
configured transport as a method on the module. LMCache calls it before the
request server starts.

For gRPC, register generated services against the concrete ``grpc.Server``:

.. code-block:: python

   from my_package.protos import echo_pb2_grpc


   class MyServerModule(DiscoverableModule):
       # regular module contract omitted

       def register_grpc_services(self, server) -> None:
           echo_pb2_grpc.add_EchoServiceServicer_to_server(
               MyEchoServicer(),
               server,
           )

With ``--grpc-server-mode native`` the server is still a ``grpc.Server``, but
it sits behind the native front on a private Unix socket: ``context.peer()``
in its handlers is that socket, not the client.

For ZMQ, register or start a package-owned ZMQ service against the concrete
``MessageQueueServer``:

.. code-block:: python

   class MyServerModule(DiscoverableModule):
       # regular module contract omitted

       def register_zmq_services(self, server) -> None:
           self.echo_service = MyZmqEchoService(
               context=server.ctx,
               bind_url="tcp://127.0.0.1:6001",
           )
           self.echo_service.start()

If a ZMQ registrar starts a sidecar socket, the module's ``close()`` method
must stop that sidecar. Do not use
``register_zmq_services`` to add ``HandlerType.BLOCKING`` handlers to the core
``MessageQueueServer`` unless the registrar also assigns an executor with
``add_normal_thread_pool`` or ``add_affinity_thread_pool`` before startup.

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

Client-affinity calls
---------------------

Mark a handler with ``requires_client_affinity=True`` to run it on the
calling client's affinity thread, the thread that also serves that client's
store and retrieve requests. Clients call it with
``server_module_affinity_call``:

.. code-block:: python

   class MyServerModule(DiscoverableModule):
       @server_module_handler("my_package.load", requires_client_affinity=True)
       def load(self, payload: bytes) -> bytes:
           ...


   response = client.server_module_affinity_call(
       ServerModuleCallRequest(method="my_package.load", payload=payload)
   ).result(timeout=5)

The transport queues the call in order with the client's other affinity
requests: it starts after an earlier store or retrieve from the same client and
holds back later ones until it returns, so keep it short. A method is served
by exactly one of ``server_module_call`` and ``server_module_affinity_call``;
calling it through the other returns an error that names the right RPC.

Example: layer-wise KV retrieval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A layer-wise retrieval plug-in lets the engine start computing early layers
while later layers are still being loaded. Once per engine instance, a unary
call sets up a pool of device events. Each retrieve is then one affinity call,
so its copies run in order with the client's other store and retrieve
requests. Progress frames travel on a channel the plug-in owns, such as a
ZMQ PUB socket it binds itself:

.. code-block:: text

   vLLM worker (layer-wise connector)        LMCache MP server
   ==================================        ==================================

   Once per engine instance, after register_kv_cache:
   server_module_call(
     "layerwise.register_events",
     instance ID)               ---------->  plug-in creates a pool of device
                                             IPC events for the instance
                                <----------  batch size + pool event handles
   open the pool handles once,
   subscribe to the plug-in's
   frame channel

   Per retrieve request:
   server_module_affinity_call(
     "layerwise.retrieve",
     request ID, key,
     instance ID,
     GPU block IDs)             ---------->  ExtServerModuleRouter
                                               -> plug-in handler, on the
                                                  client's affinity thread,
                                                  per batch of layers:
                                                  - enqueue H2D copy + scatter
                                                    for layers L..L+B-1
                                                  - record pool event i
                                                  - publish frame
                                                    (request ID, L, B, i)

   For each frame:              <---------   plug-in frame channel
     GPU stream waits on pool event i
     compute layers L..L+B-1
     (later batches keep loading)

   Call result:                 <----------  success=True once every copy is
                                             enqueued (success=False if the
                                             handler raised)
   On failure or timeout, recompute
   the affected blocks.

Core provides plug-in loading, handler registration through
``server_module_handler``, routing of both envelope RPCs through
``ExtServerModuleRouter``, and running the affinity call on the client's
affinity thread on either ``--transport``. The plug-in owns everything
layer-specific: the frame format and channel, the event pool, and the
worker-side connector that waits on each event.

Each frame means the copies for that batch are *enqueued*, not finished; the
worker learns they have landed only through the pool event. Device IPC event
handles only work within one host, so the worker and the server must run on
the same node.


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
