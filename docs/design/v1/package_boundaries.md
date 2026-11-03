# Client/server package skeleton and incremental migration

## Motivation

LMCache currently ships inference-engine adapters, client transport, server
execution, storage backends and control-plane services from one distribution.
That makes an engine worker install and upgrade code and dependencies that it
does not execute. The long-term goal is to release an additional
`lmcache-client` wheel that contains the canonical client implementation but
omits the cache server, frontend and server control planes.

"Lightweight" means a smaller source and dependency closure than the full
server distribution. It does not promise a pure-Python or torch-free wheel:
engine adapters still need platform abstractions and, where applicable, native
client-side transfer support.

The source split comes first so that the client wheel can select stable package
roots instead of maintaining a second implementation. Once the boundary is in
place, normal client changes should continue to land once in the canonical
source tree; packaging should only select and validate that source. This keeps
the lightweight package close to the full package and minimizes ongoing
packaging-specific changes.

## Status of this PR

This PR introduces import-free package markers and this migration plan.
Existing implementations, imports, CLI entry points, wire formats, tests and
build configuration remain in place. The new packages do not re-export existing
implementations and do not yet provide a client API or a client wheel.

The skeleton establishes agreed destinations before implementation moves in
small, independently reviewable PRs. Every follow-up should move one coherent
responsibility and keep existing consumers working throughout the transition.

## Target ownership

```text
lmcache/
  integration/                    client-owned engine adapters; path retained
  sdk/                            client-owned public SDK; path retained
  v1/
    client/                       worker configuration and client workflows
      transport/                  outbound connections and client polling
    common/                       pure data/helpers used by both endpoints
    platform/                     device/platform vertical; path retained
    transport/                    shared protocols, schemas, framing and codecs
    server/
      cache_server/               `lmcache server` execution and storage
      frontend/                   LMCache fleet frontend and static UI
      mp_coordinator/             MP fleet coordination service
      memory_orchestrator/        memory allocation/readiness service
      transport/                  reusable inbound transport infrastructure
```

These are target responsibilities, not claims that code has already moved.
The existing `multiprocess`, `distributed`, `lmcache_frontend`,
`mp_coordinator`, `memory_orchestrator` and other packages keep their current
responsibilities until the corresponding migration PR lands.

### Core and client-side vertical packages

| Destination | Ownership rule |
| --- | --- |
| `client` | Worker registration, subscriptions, transfer lifecycle and other inference-worker workflows. |
| `common` | DTOs, key/layout descriptions and pure helpers actually needed by both endpoints. It must not become a home for platform or endpoint implementation. |
| `transport` | Contracts, operation IDs, protobuf schemas/bindings, framing, shared serialization and error envelopes. |
| `client/transport` | ZMQ/gRPC outbound calls, pending requests, polling and connection cleanup. |
| `platform` | Device detection, device operations, IPC primitives and platform plugins. Both client and server may consume it, but it stays a separate vertical package because it is not pure endpoint-agnostic data. The client wheel includes the client-required platform closure. |
| `integration` | vLLM, SGLang, TensorRT-LLM and other inference-engine adapters. This is client-owned and included in the client wheel; standalone server components must not depend on it. Its public `lmcache.integration.*` paths stay stable instead of moving below `v1.client`. |
| `sdk` | The public programmatic client API. It remains at `lmcache.sdk`, uses the same canonical client transport/transfer code, and is included in the client wheel after server-only dependencies are removed from its import closure. |

Wheel inclusion and directory nesting are deliberately different concepts.
Keeping `platform`, `integration` and `sdk` at their established paths avoids a
large import migration while still giving the future client manifest explicit
ownership rules.

### Server service placement

The word "server" covers four independently runnable products. They must not be
collapsed into one flat package or moved wholesale when a current directory
contains both endpoint roles.

| Current surface | Canonical server destination | Parts that move elsewhere |
| --- | --- | --- |
| `lmcache server`, mainly `v1/multiprocess` plus `v1/distributed` | `server/cache_server` for process composition, request execution, storage ownership, eviction and its HTTP API | Worker lifecycle to `client`; wire contracts to `transport`; outbound calls to `client/transport`; reusable listeners/dispatch to `server/transport`; platform code stays in `platform` |
| `lmcache/lmcache_frontend` | `server/frontend` for the fleet proxy, heartbeat service, configuration and static assets | Shared request/response contracts go to `transport` or `common`; the old import path remains a compatibility facade until separately deprecated |
| `v1/mp_coordinator` | `server/mp_coordinator` for the app, controllers, ingest loops, persistence, registries, views and HTTP routes | Coordinator DTOs/contracts go to `transport` or `common`; outbound coordinator clients such as blend lookup go to `client/transport` or a client workflow package |
| `v1/memory_orchestrator` | `server/memory_orchestrator` for the gRPC service, state machine, allocation authority and server entry point | Protobuf/API/codec contracts go to `transport`; the outbound orchestrator client goes to `client/transport` |

Service-specific FastAPI routes, gRPC servicers and composition stay with their
service. `server/transport` is only for reusable listener, dispatch, handler
scheduling and server-side connection lifecycle; it must not become a second
home for service business logic. Similarly, `transport` owns endpoint-neutral
wire contracts, not sockets or execution pools.

Placement follows the responsibility of a symbol, not the name of its current
file. Inspect callers before moving it. A mixed module must be split by symbol;
moving it wholesale into `common` would preserve the unwanted dependency.

## Dependency direction

```text
integration ------> client ------> common
       |               |
       |               +---------> client/transport ----+
       v                                           |     |
   platform <--------------------------------------+     v
       ^                                           transport ---> common
       |                                                 ^
sdk ---> client/transport                               |
                                                         |
server services ---> server/transport -------------------+
       |              |
       +------------> common
       +------------> platform (only when the service needs a device primitive)
```

`common` and public `transport` must not import endpoint implementations.
`client`, `integration` and `sdk` must not import `server`, including through
lazy imports, package initializers or type-only dependencies. A server that
initiates a peer RPC may reuse an outbound implementation from
`client/transport`; this does not permit a dependency from the client back to
server storage or orchestration. Server services must not import
`lmcache.integration`.

These rules apply as new implementations enter the skeleton. This PR does not
add a global gate against the legacy tree, which still contains mixed modules.
Each migration PR should add a focused check for the boundary it establishes;
the complete client source-selection gate comes before wheel extraction.

## Client wheel and SDK decision

The release plan is one additional `lmcache-client` distribution containing the
canonical client code, `lmcache.integration`, the client-required
`lmcache.v1.platform` closure and `lmcache.sdk`. The full `lmcache` distribution
continues to include both endpoints and all server products.

There will not be separate `lmcache-client` and `lmcache-client-with-sdk`
wheels. Python extras can add dependencies but cannot conditionally add Python
files to an already-built wheel, while two distributions that both install the
same `lmcache` modules create ambiguous file ownership, uninstall and upgrade
behavior. A separate `lmcache-sdk` distribution should be considered only if
the SDK later needs an independent compatibility/release lifecycle and can own
a non-overlapping package surface. That is outside this migration.

Because `lmcache` and `lmcache-client` expose the same import namespace, they
are alternative distributions for an environment, not packages to co-install.
Release validation must test clean installation, replacement/uninstallation
behavior and an explicit supported-version matrix against the server.

Engine libraries remain optional integration dependencies: installing the
client wheel must not pull every supported inference engine. Their imports and
dependency extras must stay isolated so that selecting one integration does not
make the others mandatory.

## Packaging without modifying the full-package build

The packaging PR must add a dedicated, version-controlled client build surface;
it must not rewrite or temporarily patch the root `pyproject.toml`, `setup.py`,
requirements files or release scripts. In particular, release automation must
not use `sed`, overwrite-and-restore steps or a dirty Git worktree to switch
between full and client builds.

The intended build is:

1. Add static client packaging files under a dedicated build directory: its
   own `pyproject.toml`/backend configuration, dependency metadata, entry points,
   package-data rules and an explicit source allowlist/denylist.
2. Have one checked-in build command create a fresh temporary staging root and
   copy only the allowlisted canonical sources into their existing `lmcache.*`
   paths. Copy the static client packaging files without generating or editing
   their contents. Build tools must run from this staging root so every
   `package_dir` is inside the build project.
3. Derive the release version once from the exact tag/revision and pass it
   through the build backend's supported environment/version interface. With
   setuptools-scm, use
   `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE_CLIENT=<release-version>` in the
   staging build. Do not write a temporary version into tracked source files.
4. Generate any client-required protobuf bindings or native build artifacts
   only inside the staging/build directories. The dedicated backend must invoke
   client-specific generators directly instead of importing or patching the
   full-package `setup.py`.
5. Keep client dependencies and optional engine extras in the static client
   metadata. Do not install server web, storage, coordinator or orchestration
   dependencies merely because they remain requirements of the full wheel.
6. Run `python -m build` and `cibuildwheel` against the staging project itself,
   then audit the built wheel's file list, METADATA, entry points, package data
   and native libraries. Fail on any `server`, frontend, server coordinator,
   server memory-orchestrator, storage-backend or server-CLI leak.
7. Install the artifact into a clean client environment with no source checkout
   and no preinstalled full `lmcache`; install the full wheel in a separate
   server environment. Exercise imports, SDK calls, each supported engine
   integration, transport compatibility and a real client/server request.

The initial allowlist is defined only after the migrations make the dependency
closure explicit. It is expected to include `client`, `common`, `transport`,
`platform`, `integration`, `sdk` and a small set of proven support modules; it
must never fall back to broad `lmcache*` package discovery. Built-wheel contents
and clean-environment behavior, not the distribution name, prove that the wheel
is lightweight.

## Follow-up PR sequence

The rows below describe future PRs, not additional changes included here.
Split a row further when it contains more than one reviewable responsibility.

| PR slice | Change | Compatibility and verification |
| --- | --- | --- |
| 1. Shared data | Extract one group of DTOs or layout/key descriptions to `common` | Keep old imports as explicit re-exports of the same objects; check identity and serialization |
| 2. Shared protocol | Extract RPC contracts, then shared codec/schema pieces into `transport` in separate PRs | Preserve operation IDs, protobuf service names and wire bytes; test both endpoint consumers and binding generation |
| 3. Endpoint transport | Split ZMQ client polling from server execution; handle gRPC ownership in a separate PR | Keep old construction paths working; test request/response, streaming, failure propagation and cleanup |
| 4. Client workflows | Move one worker-side registration/transfer/subscription responsibility at a time | Update its callers with compatibility imports retained; prove the client closure does not import server implementation |
| 5. Client vertical audit | Remove server-only dependencies from `integration`, `platform` and `sdk` import closures without changing their public paths | Test isolated imports and supported engine/SDK behavior with optional dependencies absent |
| 6. Cache server | Move one `lmcache server` execution or storage responsibility into `server/cache_server` | Preserve CLI/plugin entry points and run focused service/storage regressions |
| 7. Other server products | Migrate frontend, MP coordinator and memory orchestrator one product and responsibility at a time | Preserve old entry/import paths; split each product's contracts and outbound clients before its server implementation |
| 8. Client wheel | Add the static staging build, exact manifest and release job after the source boundary is complete | Audit the wheel and run separate-environment client/server compatibility tests |

Shared data and protocol extraction precede the consumers that need them.
Client and server workflow migrations can then proceed independently where
their prerequisites are already merged.

## Checklist for each migration PR

1. Identify the implementation's callers, dependencies and public import paths.
   Name the single responsibility being moved in the PR description.
2. Put the canonical definition in its destination and retain explicit legacy
   re-exports where compatibility is required. Do not duplicate classes or
   introduce a global import hook. Avoid an eager aggregate export that pulls
   server code into the client import chain.
3. Keep function signatures, CLI entry points, wire contracts and plugin hooks
   stable for that slice. Inspect persisted module-qualified names, trace
   records and serialized Python values before changing their defining module;
   provide a compatibility strategy and tests if they are affected.
4. Update in-tree consumers, mirrored design docs and any affected generation,
   ownership or build paths in the same PR. Add boundary/identity tests alongside
   the relevant behavioral tests. Select checks in proportion to the slice.
5. Verify old and new imports coexist, the established dependency direction
   holds, and supported client/server version combinations still work. Each PR
   must remain independently mergeable; reverting a slice must not require
   unrelated migrations to land.

Remove compatibility paths only after their consumers have migrated, in an
explicitly reviewed cleanup with a documented deprecation/version policy. The
final wheel gate must test the built distribution, not only source imports.

## Review and validation for the skeleton

Review the package docstrings and this plan. Package imports execute no new
implementation code; this PR changes no production call chain. Validation
should confirm that the package markers contain no executable imports and that
the diff is additive, then run the repository's required formatting/type
checks. Runtime transport/storage tests and wheel-size claims belong to the
later PRs that implement those changes.
