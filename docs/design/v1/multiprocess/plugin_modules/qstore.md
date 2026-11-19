# QStore Plugin Module

> Server side of query-tensor transfer: `QStoreModule` in
> `lmcache/v1/multiprocess/plugin_modules/qstore.py`. An opt-in module that
> is never auto-discovered.

## What it is

`QStoreModule` registers a worker's paged Q ring buffer and stores its blocks
through the same copy path as the LMCache-driven KV transfer, keyed under the
query-specific model name `<model>##query`. The worker-side producer is the
connector's `QRingBufferCapture`; see [`sdk/qringbuffer.md`](../../../sdk/qringbuffer.md).

## Why it lives outside the discovery root

Built-in modules are discovered by scanning `lmcache.v1.multiprocess.modules`
recursively (`engine_module.discover_modules`). Query-tensor transfer is an
experimental, opt-in capability, so its module lives in
`lmcache.v1.multiprocess.plugin_modules`, which that scan never visits. The
server composes it only when the operator names it:

```bash
lmcache server \
    --supported-transfer-mode auto \
    --server-module '{"module_path": "lmcache.v1.multiprocess.plugin_modules.qstore"}'
```

`--server-module` imports the leaf module and registers the concrete
`DiscoverableModule` classes defined in it (imported dependency classes are
ignored), then the module joins the normal dependency graph and lifecycle.
There is no separate enable flag.

Nothing under `lmcache.v1.multiprocess.modules` may import this package at
module scope: discovery accepts re-exported classes, so such an import would
make the module an always-built built-in. `test_qstore_is_not_auto_discovered`
guards this.

## Construction contract

- `module_name = "qstore"`, `module_dependencies = (LMCacheDrivenTransferModule,)`.
- `create()` requires the LMCache-driven transfer module to be built. When
  `--supported-transfer-mode` is `engine_driven`, startup fails with
  "requires supported_transfer_mode='lmcache_driven' or 'auto'".
- `create()` calls `build_ctx.add_experimental(TRANSFER_QUERY)`, so the
  management module's `get_experimental()` returns `["transfer_query"]`.
- The module implements `InstanceLivenessTarget`; the management reaper
  touches and reaps its per-instance Q contexts like the KV transfer modules.

## Client handshake

The vLLM connector opts in with
`"lmcache.mp.transfer_intermediate_tensors": true`. At startup the worker
adapter calls `get_experimental`; `init_dispatcher` refuses to build the
query feature when `transfer_query` is not advertised ("Connector enables
transfer_query but server does not."), which surfaces a server started
without the plugin as a clear startup error instead of a failed store.

## Wire protocol

Unchanged by the plugin packaging: `register_q_cache`, `unregister_q_cache`
and `store_q` are first-class `RequestClient` operations (ZMQ legacy ids 3-5,
gRPC `QStoreService` in `protos/qstore_service.proto`). When the plugin is not
loaded, the gRPC transport answers those methods with `UNIMPLEMENTED` and the
ZMQ transport rejects the unknown operation.
