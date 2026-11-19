# Multi-Chunk-Size Support

## Background

The LMCache multiprocess (MP) server currently uses one global chunk size.
Initial negotiation may change that global value, which prevents one MP server
from serving inference deployments that require different chunk sizes.

`LayoutDescRegistry` already uses `(model_name, world_size)` to connect worker
registration metadata with scheduler lookups. A worker computes its KV layout
from the global chunk size during KV-cache registration and stores the result in
the registry. Because the registry already selects layouts by model and
parallelism, it can also own the metadata needed for multiple chunk sizes.

## Terminology

- **Server minimum chunk size**: The minimum configured on the MP server with
  `--chunk-size`. It remains unchanged throughout the server lifetime.
- **Client minimum chunk size**: An optional minimum configured by the client;
  `None` when it is not configured.
- **Base chunk size**: The larger of the server and client minimums.
- **Effective chunk size**: The base chunk size rounded up to the model's
  required alignment.
- **Registration key**: The `(model_name, world_size)` pair that uniquely
  identifies shared registration metadata.

## Existing Flow

The MP server does not create layouts at startup. A worker creates them when it
registers its KV cache:

```text
MP server starts
    |
    `-- ctx.chunk_size = configured server default
            |
            `-- global negotiation may change ctx.chunk_size
                    |
worker register_kv_cache()
    |
    |-- create cache_context
    |-- get_layout_desc(cache_context, ctx.chunk_size, ...)
    |       `-- compute the tensor shapes and dtypes of one KV object
    `-- LayoutDescRegistry.register(...)
            `-- store the computed layout
```

`MemoryLayoutDesc` stores shapes and dtypes but does not explicitly store a
chunk size. However, `get_layout_desc(cache_context, chunk_size, ...)` uses the
chunk size to compute those shapes, so every layout already depends implicitly
on the chunk size used to create it.

Scheduler lookup and prefetch subsequently read layouts from the registry,
while store, retrieve, and other data paths still use the global
`ctx.chunk_size` and `ctx.token_hasher`. Adding fields to the registry is
therefore insufficient by itself: the normal data paths must also select
metadata by registration key.

## Goals

- A client can configure its own minimum chunk size.
- Both the server and client chunk-size settings act as minimums.
- A client always reads the MP server minimum, whether or not it configures a
  client minimum.
- `(model_name, world_size)` uniquely determines the effective chunk size.
- One MP server can serve models that use different chunk sizes.
- The server default chunk size never changes.

## Client Configuration

An inference-engine integration can provide an optional client minimum:

```yaml
lmcache.mp.chunk_size: 512
```

- This value is a minimum requested by the client, not the exact final value.
- The client always calls `get_chunk_size()` to obtain the MP server minimum.

## Chunk-Size Resolution

The client resolves the chunk size because model block geometry and required
alignment are known by the inference engine.

The resolution flow is:

1. Read the optional client chunk-size setting.
2. Always call `get_chunk_size()` to obtain the server minimum.
3. Take the larger of the server and client minimums. If there is no client
   minimum, use the server minimum.
4. Round that value up to the model's required alignment to obtain the
   effective chunk size.
5. Ensure that the scheduler and every worker or rank use the same effective
   value.
6. Include the effective chunk size when each worker registers its KV cache.

Resolution follows these rules:

- Both configured minimums must be positive.
- The effective value must not be smaller than either configured minimum.
- A base value that is not aligned is rounded up; an aligned value is left
  unchanged.

The calculation is:

```text
base = server_minimum if client_minimum is None
       else max(server_minimum, client_minimum)
effective = round_up(base, required_alignment)
```

Examples:

| Server minimum | Client minimum | Required alignment | Effective chunk size |
|---:|---:|---:|---:|
| 256 | Not configured | 16 | 256 |
| 256 | 128 | 16 | 256 |
| 256 | 512 | 16 | 512 |
| 256 | 500 | 16 | 512 |
| 256 | 300 | 80 | 320 |
| 256 | Not configured | 80 | 320 |

## New Flow

```text
client starts
    |
    |-- client_min = optional client setting
    `-- server_min = server.get_chunk_size()  # always called
            |
            v
base = server_min if client_min is None else max(server_min, client_min)
            |
            v
effective = round_up(base, required alignment)
            |
            |-- scheduler uses it for lookup and scheduling
            `-- every worker/rank uses it for registration and transfer
                    |
                    v
register_kv_cache(
    model_name,
    world_size,
    effective_chunk_size,
    ...
)
                    |
                    v
server creates the layout and TokenHasher for the effective chunk size
                    |
                    v
LayoutDescRegistry[(model_name, world_size)]
```

The server no longer changes its default chunk size based on one model's
resolution result. The client resolves the effective value, while the server
registers and validates it and exposes the metadata required to process
requests.

## Server Registration

Extend KV-cache registration with the effective chunk size:

```text
register_kv_cache(
    instance_id,
    model_name,
    world_size,
    chunk_size,
    ...
)
```

Engine-driven registration carries the same information.

During registration, the server:

1. Uses the supplied chunk size to create the layout.
2. Creates a corresponding `TokenHasher` for that chunk size.
3. Stores both in `LayoutDescRegistry`.
4. Verifies that every rank sharing `(model_name, world_size)` uses the same
   chunk size.

## LayoutDescRegistry

The registry key remains:

```text
(model_name, world_size)
```

Each entry adds the following metadata:

```python
@dataclass
class _LayoutDescEntry:
    chunk_size: int
    token_hasher: TokenHasher
    layout_desc: MemoryLayoutDesc
    group_layout_descs: dict[int, MemoryLayoutDesc] | None
    # Other existing registration metadata.
```

For example:

```text
("model-a", 1) -> chunk_size=256
("model-a", 2) -> chunk_size=320
("model-b", 2) -> chunk_size=512
```

The server rejects a different chunk size registered under an existing
`(model_name, world_size)` key. The existing reference-count mechanism removes
the entry after the final associated worker unregisters. Once removed, a later
worker group may register the same key with a different chunk size.

The chunk size must be stored explicitly rather than inferred from
`layout_desc.shapes`. Shape semantics vary across KV formats, and lookup chunk
boundaries, token hashes, and transfer block counts all need the original chunk
size directly.

## Request Processing

Lookup, store, and retrieve use `key.model_name` and `key.world_size` to obtain
the following metadata from `LayoutDescRegistry`:

- chunk size;
- token hasher; and
- layout information.

Normal prefix-cache data paths no longer use the global values:

```text
ctx.chunk_size
ctx.token_hasher
```

A worker context may cache the registered chunk size to avoid registry lookups
on transfer hot paths.

Responsibilities are divided as follows:

- Scheduler lookup uses the registered `token_hasher` to compute chunk hashes
  and the registered `chunk_size` to convert hits into token counts.
- Prefetch uses the registered layout and token hasher.
- Worker store and retrieve use the chunk size cached in the registration
  context to calculate chunk counts, partial chunks, and blocks per chunk.
- A registration context's cached value must match the value stored under its
  registration key.
- Metrics and events use the actual effective chunk size instead of assuming a
  global chunk size.

## Server Default

The MP server's `--chunk-size` remains the server-provided minimum and never
changes in response to a model's resolved value. A client setting may raise
this lower bound but cannot reduce the effective chunk size below it.

For example:

```text
server default = 256

(model-a, 1) -> client minimum 128, alignment 16 -> use 256
(model-b, 2) -> client minimum 512, alignment 16 -> use 512
(model-c, 4) -> no client minimum, alignment 80  -> use 320
```

The server default remains 256 in all three cases.

## Scheduler and Workers

Every engine integration must make its scheduling and transfer participants
agree on the effective chunk size:

- The scheduler uses it for lookup and scheduling granularity.
- Workers use it to calculate blocks per chunk and to register and transfer KV
  data.
- An integration in which one connector performs both responsibilities still
  resolves the value once and shares it consistently across all ranks.

## Unfull-Chunk Compatibility

An unfull chunk's `ObjectKey` includes the registration's effective full chunk
size as the identity tag `chunk_size`. Full-chunk keys
remain unchanged. The tag participates in equality, hashing, L2 key encoding,
and internal wire serialization, so unfull objects created for different chunk
sizes cannot match each other or a full object with the same content hash.

For example, consider the following sequence for the same
`(model_name, world_size)` registration key:

1. Register with an effective chunk size of 512.
2. Store a 256-token prompt as an unfull chunk.
3. Unregister the final worker, then register again with a chunk size of 256.
4. Look up the same 256-token prompt, which is now one full chunk.

The first object is tagged with `chunk_size=512`, while the
second is an untagged full-chunk object. They therefore have different keys,
and the lookup cannot retrieve an object whose size and layout belong to the
previous registration.

**Upgrade note:** Unfull chunks stored by an older LMCache version do not have
the `chunk_size` identity tag and can still collide with full chunks after the
effective chunk size changes. Before the first chunk-size change after an
upgrade, clear the existing L2 cache. If the L2 cache cannot be cleared, keep
the previous chunk size until its old objects have been removed.

## Coordinator and CacheBlend

Coordinator and CacheBlend still depend on one global chunk size and do not yet
support model-specific effective chunk sizes. While either feature is enabled:

- the MP server requires each registration's effective chunk size to equal its
  configured `--chunk-size` and rejects mismatches;
- an MP server registering with Coordinator continues to report its configured
  `--chunk-size` through the `chunk_size` field in `POST /instances`;
- Coordinator returns HTTP 409 when a server reports a value different from
  the Coordinator's configured global chunk size, so the mismatched server is not
  registered; and
- deployments must configure the MP servers and Coordinator with one chunk size
  that satisfies every connected model.

The per-server registration check is not sufficient by itself: it only binds an
engine to its own MP server's global value. The Coordinator registration check
must remain until Coordinator hashing, pinning, routing, and blend matching can
select chunk-size metadata by registration key.

A later extension can make Coordinator and CacheBlend select chunk-size
metadata by `(model_name, world_size)` as well.
