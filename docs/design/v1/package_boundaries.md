# Client/server package skeleton and incremental migration

## Status of this PR

This PR introduces six import-free package markers and this migration plan.
Existing implementations, imports, CLI entry points, wire formats, tests and
build configuration remain in place. The new packages do not re-export existing
implementations and do not yet provide a client API or a client wheel.

The skeleton establishes agreed destinations before implementation moves in
small, independently reviewable PRs. Every follow-up should move one coherent
responsibility and keep existing consumers working throughout the transition.

## Target ownership

```text
lmcache/v1/
  client/                 inference-worker configuration and client workflows
    transport/            outbound connections, RPC clients and client polling
  common/                 data types and helpers needed by both endpoints
  transport/              shared protocols, schemas, framing and codecs
  server/                 request execution, storage and server control plane
    transport/            listeners, dispatch, handlers and execution resources
```

These are target responsibilities, not claims that code has already moved.
The existing `multiprocess`, `distributed`, `platform` and other packages keep
their current responsibilities until the corresponding migration PR lands.

| Destination | Include when migrating | Keep elsewhere |
| --- | --- | --- |
| `client` | Worker configuration, registration, subscriptions and worker transfer lifecycle | Server composition, cache storage and eviction |
| `common` | DTOs, key/layout descriptions and pure helpers actually used by both endpoints | Endpoint-specific implementation or optional backend initialization |
| `transport` | RPC contracts, protobuf schemas/bindings, framing, shared serialization and error envelopes | Client channels/polling and server listeners/thread pools |
| `client/transport` | ZMQ/gRPC outbound calls, pending requests and connection cleanup | Request execution and storage management |
| `server` | Cache-server orchestration, storage, coordination and server-only device resources | Worker integration or client-only transfer workflows |
| `server/transport` | Listening sockets, request dispatch, handler scheduling and resource cleanup | Protocol definitions duplicated from the shared transport package |

Placement follows the responsibility of a symbol, not the name of its current
file. Inspect callers before moving it. A mixed module must be split by symbol;
moving it wholesale into `common` would preserve the unwanted dependency.
For example, a shared-memory slot descriptor can be shared while the worker's
pool lifecycle stays client-side. Likewise, shared ZMQ encoding is separate
from client polling and server execution pools.

## Dependency direction

```text
client ---------------------> common
client/transport ------------> transport ----> common
server ---------------------> common
server/transport ------------> transport ----> common
server peer-RPC caller ------> client/transport
```

`common` and public `transport` must not import client/server implementations.
`client` must not import `server`, including through lazy imports, package
initializers or type-only dependencies. A server initiating a peer RPC may
reuse the outbound client implementation. This does not permit a dependency
from the client back to server storage or orchestration.

These rules apply as new implementations enter the skeleton. This PR does not
add a global gate against the legacy tree, which still contains mixed modules.
Each migration PR should add a focused check for the boundary it establishes;
the complete client source-selection gate comes before wheel extraction.

## Follow-up PR sequence

The rows below describe future PRs, not additional changes included here.
Split a row further when it contains more than one reviewable responsibility.

| PR slice | Change | Compatibility and verification |
| --- | --- | --- |
| 1. Shared data | Extract one group of DTOs or layout/key descriptions to `common` | Keep old imports as explicit re-exports of the same objects; check identity and serialization |
| 2. Shared protocol | Extract RPC contracts, then shared codec/schema pieces into `transport` in separate PRs | Preserve operation IDs, protobuf service names and wire bytes; test both endpoint consumers and binding generation |
| 3. Endpoint transport | Split ZMQ client polling from server execution; handle gRPC ownership in a separate PR | Keep old construction paths working; test request/response, streaming, failure propagation and cleanup |
| 4. Client workflows | Move one worker-side registration/transfer/subscription responsibility at a time | Update its callers with compatibility imports retained; test without importing server implementation |
| 5. Server workflows | Move one server-only module or storage responsibility at a time | Preserve CLI/plugin entry points; run its service/storage regressions without changing protocol behavior |
| 6. Client wheel | Define the source/dependency/native-extension manifest after the source boundary is complete | Install the built wheel in a clean environment without server source/dependencies and run real client/server compatibility tests |

Shared data and protocol extraction precede the consumers that need them.
Client and server workflow migrations can then proceed independently where
their prerequisites are already merged. Package names and release tooling are
decisions for the wheel PR, not this skeleton.

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
explicitly reviewed cleanup with a documented deprecation/version policy.
The final wheel gate must test the built distribution, not only source imports.

## Review and validation for the skeleton

Review the six package docstrings and this plan. Package imports execute no
new implementation code; this PR changes no production call chain. Validation
should confirm that the package markers contain no executable imports and that
the diff is additive, then run the repository's required formatting/type checks.
Runtime transport/storage tests and wheel-size claims belong to the later PRs
that implement those changes.
