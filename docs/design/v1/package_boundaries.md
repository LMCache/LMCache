# Client package boundaries for an additional wheel

## Motivation

LMCache currently publishes one complete `lmcache` wheel. Inference workers
therefore install server implementations and server-only dependencies that they
do not execute. The goal is to publish an additional `lmcache-client` wheel
from the same canonical source tree while leaving the existing full wheel
unchanged.

"Lightweight" means a smaller, explicitly audited source and dependency closure
than the full distribution. It does not promise a pure-Python or torch-free
client because integrations may still require platform abstractions and native
client-side transfer support.

## Decision

This effort introduces client-facing source boundaries only:

```text
lmcache/
  integration/                 client-owned engine adapters; path retained
  sdk/                         public client SDK; path retained
  v1/
    client/                    inference-worker workflows
      transport/              outbound connections and client polling
    common/                    endpoint-neutral data and pure helpers
    platform/                  device/platform vertical; path retained
    transport/                 shared wire contracts, schemas and codecs
```

Existing server implementations stay in their current packages. This plan does
not introduce an `lmcache.v1.server` hierarchy, relocate server code or create a
server-only distribution. A later change may revisit server source organization
if it has a concrete need independent of the client wheel.

The two distributions have different scopes:

| Distribution | Scope |
| --- | --- |
| Existing `lmcache` wheel | Remains the complete product with the same build path, package contents, public imports, CLI entry points and supported features. |
| Additional `lmcache-client` wheel | Selects the client-required subset of the same canonical sources and excludes server implementations and server-only dependencies. |

Because both distributions expose the `lmcache` import namespace, they are
alternatives for an environment rather than packages to co-install.

## Status of this PR

This PR adds import-free markers for `client`, `client.transport`, `common` and
`transport`. It does not move implementations, change imports, modify build
configuration or produce the client wheel. The markers establish destinations
for small follow-up migrations while existing behavior remains unchanged.

## Ownership rules

| Package | Responsibility |
| --- | --- |
| `lmcache.v1.client` | Worker registration, subscriptions, transfer lifecycle and other inference-worker workflows. |
| `lmcache.v1.client.transport` | Outbound RPC calls, pending requests, polling and client-side connection cleanup. |
| `lmcache.v1.common` | DTOs, key/layout descriptions and pure helpers needed by both endpoints. It must not depend on endpoint implementations or become a home for platform code. |
| `lmcache.v1.transport` | Operation IDs, protobuf schemas/bindings, framing, shared serialization and error envelopes used by both endpoints. It must not own sockets, listeners, request execution or service business logic. |
| `lmcache.v1.platform` | Device detection, device operations, IPC primitives and platform plugins. Its existing path remains stable; the client wheel includes only its proven client-required closure. |
| `lmcache.integration` | Inference-engine adapters. Their existing public paths remain stable and supported integrations are included without making every engine dependency mandatory. |
| `lmcache.sdk` | Public programmatic client API. Its existing path remains stable and it is included after server-only dependencies leave its import closure. |

Directory nesting and wheel inclusion are deliberately different concepts.
Keeping `platform`, `integration` and `sdk` at their established paths avoids an
unnecessary import migration while the client build selects an exact allowlist.

## Dependency direction

```text
integration ----> client ----> common
      |              |
      |              +------> client/transport ----> transport ----> common
      v                              |
  platform <-------------------------+
      ^                              |
      +------------------------------ sdk

existing server packages ----------> transport/common as shared contracts move
```

`common` and `transport` must not import endpoint implementations. Client-side
packages must not import server implementations, including through package
initializers, lazy imports or type-only dependencies. Existing mixed modules
remain unchanged until a focused migration establishes and tests the relevant
boundary.

## Client wheel build

The root build remains the source of the full `lmcache` wheel. The client wheel
will use a separate, checked-in build path and must not patch or temporarily
rewrite the root `pyproject.toml`, `setup.py`, requirements or release scripts.

The packaging follow-up will:

1. Define static client metadata, dependencies, entry points and package-data
   rules in a dedicated build directory.
2. Copy an explicit allowlist of canonical sources into a fresh staging root
   while preserving their `lmcache.*` import paths.
3. Derive the version from the exact release revision and pass it through the
   build backend's supported version interface.
4. Generate client-required bindings or native artifacts only inside staging
   and build directories.
5. Build and audit the client wheel's files, metadata, entry points, package
   data and native libraries, failing on server implementation leaks.
6. Install the client and full wheels in separate clean environments. Verify
   client imports, supported integrations, SDK behavior, transport
   compatibility and a real client/server request, while confirming that the
   full wheel retains its existing surface.

Broad `lmcache*` discovery is not proof of a client-only artifact. The exact
manifest and clean-environment behavior are release gates.

## Follow-up sequence

Each follow-up moves one coherent responsibility and remains independently
mergeable:

1. Move shared DTO or key/layout slices into `common`, retaining explicit
   identity-preserving compatibility exports where required.
2. Move shared schemas, operation IDs and codecs into `transport`, preserving
   service names and wire bytes.
3. Move outbound endpoint implementations into `client.transport`, keeping
   listeners, handlers and execution resources in their existing server-side
   packages.
4. Move client workflows and audit the `platform`, `integration` and `sdk`
   import closures without changing their established public paths.
5. Add the static client build, exact manifest, artifact audit and
   separate-environment end-to-end validation.

Every migration must inspect callers, retain compatibility paths where needed,
and add focused identity, boundary and behavioral tests. Persisted
module-qualified names, wire contracts, CLI entry points and plugin hooks must
remain stable unless a separate change defines and tests their migration.

## Review and validation for this skeleton

Review the four package markers and the ownership rules above. The markers
contain no executable imports, and this PR changes no runtime call chain or
packaging behavior. Validation therefore checks marker contents, diff hygiene,
documentation rendering and the repository's required formatting and type
checks. Runtime, wheel-content and end-to-end claims belong to the follow-up
PRs that implement them.
