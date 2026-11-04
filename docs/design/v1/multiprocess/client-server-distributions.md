# Client and server distributions of the `lmcache` package

## Motivation

An inference-engine worker that talks to an LMCache MP server needs only the
client side of the code: the engine connectors, the request clients, the
worker-local transfer runtime and the contracts shared with the server.
Installing `lmcache` today also installs the server, the storage backends,
the in-process runtime and their dependencies into every engine image.

This design builds two distributions from the same source tree:

| Distribution | Ships | Installed by |
|---|---|---|
| `lmcache` (client) | files listed in `setup_extensions/lmcache_client_files.txt`, the `lmcache_native` extension, generated gRPC bindings and `_version.py` | `pip install lmcache` (add `[grpc]` for the gRPC transport, `[blake3]` for the default KV-event hash) |
| `lmcache-server` (full runtime) | every other file of the `lmcache` package, the storage and GPU extensions, the `lmcache` and `lmcache_controller` commands | `pip install "lmcache[server]"` or `pip install lmcache-server` |

Both install into the same `lmcache` import package. No file belongs to both,
so they can be installed together and either one can be removed without
deleting the other's files. `lmcache-server` requires `lmcache[grpc]` at its
own version, because the server imports the client's modules in-process.

The names are a candidate for review: the roadmap item calls the client
`lmcache-client`. See "Naming" below.

## Ownership rules

- `setup_extensions/distributions.py` decides which distribution ships a file
  under `lmcache/`: the client owns what `lmcache_client_files.txt` lists, the
  server owns the rest. `setup.py` filters `build_py` output with it.
- Shared contracts have one definition, owned by the client: RPC
  descriptors, DTOs, layouts, codecs, `custom_types`, the root package and the
  platform singletons. The server imports them.
- `setup_extensions/split_contexts.json` assigns every file outside the
  package (C++ sources, requirement files, pyproject files) to the client, the
  server, or both as shared build tooling. `.github/scripts/split_export.py`
  exports each distribution's source context from one commit; the server
  context reads the few client files it needs to build (KV format headers,
  the proto generator, `common.proto` and `p2p_service.proto`) from the client
  sdist.

## Code changes the split needs

Four import edges made a client module load server code:

| Cut | Before | After |
|---|---|---|
| Transfer-channel addresses | `transport/base.py` imported `TransferChannelAddress` from `v1/distributed/transfer_channel`, whose package `__init__` loads the channel factory and implementations | the DTOs live in `v1/multiprocess/transfer_channel_types.py`; the old module re-exports them |
| ZMQ message queue | `zmq_impl/mq.py` held both `MessageQueueClient` and `MessageQueueServer` with its handler executor | the server half moved to `zmq_impl/server_mq.py`; `mq.__getattr__` resolves the old server names on access |
| Timeout error | `futures.py` imported `LMCacheTimeoutError` from `mp_observability`, which loads the event bus | `v1/multiprocess/client_errors.py` defines the error and an observer hook; `mp_observability.errors` registers the observer and re-exports the class; `init_event_bus` registers it too |
| Block-skip helper | engine-driven scatter imported `recalculate_blocks_to_skip` from the server's `object_group_transfer` | the helper lives in `v1/multiprocess/block_id_utils.py` |

`lmcache.utils.TTLCache` imports `cachetools` when an instance is built, so
the client does not declare `cachetools`; only server storage backends build
one.

## Allowed client-to-server references

`.github/scripts/check_split_boundary.py` scans every client module in the
built client wheel for imports of server-owned modules, including function-level,
`TYPE_CHECKING`-guarded and `importlib` imports.
`.github/scripts/split_boundary_allowlist.json` lists the allowed ones, each
with a reason:

- type-only imports;
- `mq.__getattr__`, which imports `server_mq` only when an old server name is
  accessed;
- the built-in `DeviceSpec.create_cache_context` hooks, which only the
  server's cache-context factory calls;
- the lazy imports of `lmcache.cuda_ops` and `lmcache.xpu_ops`. These GPU
  extensions are built by `lmcache-server`. Without them the device ops log a
  warning and use the torch implementation. CUDA and NPU workers default to
  the lmcache-driven transfer mode, whose copies run in the server; XPU and
  other engine-driven devices use the fallback in a client-only install.
  Whether the fallback is correct and fast enough on GPUs is not yet
  verified, so XPU deployments should keep the full install for now.

## Checks

`.github/workflows/split_packaging.yml` runs `split_packaging_gate.sh`:

1. export and build each distribution's sdist and wheel from its own context;
2. compare the wheels with the base commit's single wheel: no path in both,
   nothing lost, new paths only where git added files;
3. verify each context file's origin and the client sdist the server used;
4. run the boundary check against the allowlist;
5. install `lmcache[grpc]` alone and run the probes in
   `.github/scripts/split_probes/` with an import hook that refuses any module
   the client wheel does not ship.

Editable installs map the whole tree, so they cannot show ownership.
`.github/scripts/install_lmcache_editable.sh` installs both distributions in
editable mode for development and CI.

## Compatibility

- `pip install -U lmcache` from a monolithic release installs only the
  client: the server modules and the `lmcache` command go away. Server and
  in-process users need `lmcache[server]`.
- The pre-split import paths of the moved names keep working when both
  distributions are installed. In a client-only install the old server-side
  paths do not exist.
- The two distributions must have the same version. This constrains one
  Python environment only; there is no protocol version check between a
  client and a server in different environments yet.

## Naming

| Option | `pip install lmcache` means | Known cost |
|---|---|---|
| `lmcache` client + `lmcache-server` (this design) | the client | changes what existing installs get |
| `lmcache-client` + `lmcache` full runtime depending on it | unchanged | a one-step `pip install -U lmcache` from a monolithic release removes most client files, because pip uninstalls the old distribution after installing the new dependency; needs a documented two-step upgrade |
| same names, client files under a private directory added by a `.pth` | unchanged | `.pth` files are skipped with `python -S`, `pip install --target`, zip apps and PyInstaller |

`check_split_wheels.py` and `check_split_boundary.py` accept `--layout A|B|H`
for these three options; the build files implement only the first.
