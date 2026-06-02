# MP Coordinator

The mp coordinator is a standalone process that coordinates LMCache
multi-process (mp) cache servers running across nodes as a fleet. This document
describes the backbone: transport, the pluggable controller seam, the instance
registry, lifecycle hooks, and the concurrency contract. Domain capabilities
(quota reconcile, blend-lookup routing, KV-op fan-out) are not implemented yet;
they are added as new controllers that plug into the same seam.

Code: `lmcache/v1/mp_coordinator/`.

## Why

mp servers are independent today: quota is per-instance and in-memory, there is
no cross-node token-match routing for model replicas, and KV operations are
local. The coordinator is the fleet-level component those features will hang
off. This PR ships only the framework plus registration so future work plugs in
without re-architecting.

## Transport

ZMQ, reusing `lmcache/v1/rpc_utils.py`. The coordinator binds three sockets:

| Socket | Pattern | Purpose |
| --- | --- | --- |
| `pull_url` | PULL | fire-and-forget messages from mp servers (deregister) |
| `reply_url` | ROUTER | request/reply (register) |
| `heartbeat_url` | ROUTER | dedicated heartbeat request/reply |

Server-initiated push (the channel for future quota broadcast / KV-op fan-out)
uses a per-instance REQ socket the coordinator opens at registration, connected
back to the mp server's control REP socket. `CommandSender` (`command.py`) sends
opaque `bytes` over these sockets; each controller encodes/decodes its own
message types around it.

```
            mp server (CoordinatorClient)        MP coordinator
  bind REP  control  <───────────────── REQ per-instance (push commands)
  PUSH      ──────────────────────────> PULL   (deregister)
  REQ       ──────────────────────────> ROUTER (register)
  REQ       ──────────────────────────> ROUTER (heartbeat)
```

ZMQ chosen over HTTP because the control plane needs server→mp push, which ZMQ
does natively; HTTP would force polling or websockets. No TLS/auth is included —
run inside a trusted network or add ZMQ CURVE/ZAP for untrusted links.

## Messages (`message.py`)

msgspec tagged structs, decoded via the `CoordMsg` union, classified by channel
so the manager routes by socket:

- `PushMsg` — fire-and-forget, arrives on PULL.
- `ReqMsg` / `ReqRetMsg` — request and reply, arrive/return on ROUTER.

Backbone messages: `RegisterMsg`/`RegisterRetMsg`, `DeregisterMsg`,
`HeartbeatMsg`/`HeartbeatRetMsg`, `ErrorMsg`.

## Controller seam (`controllers/base.py`)

A `Controller` declares the message types it handles via `push_handlers()` and
`req_handlers()`. `MPCoordinatorManager` merges every controller's
declarations into two dispatch tables at startup and routes each decoded message
by `type(msg)`. There is no `isinstance` chain to edit when adding a capability.

Controllers receive a `ControllerContext` at `post_init` carrying the shared
`InstanceRegistry`, the `CommandSender`, the `LifecycleHooks`, the ZMQ context,
and a `get_controller(type)` accessor. Controllers reach collaborators only
through the context — they never import one another.

### Adding a controller (the extension point)

1. Add message types in `message.py` (subclass `PushMsg` / `ReqMsg` /
   `ReqRetMsg`) and list them in the `CoordMsg` union.
2. Add `controllers/<name>.py` with a `Controller` subclass declaring handlers
   for those types; wire collaborators in `post_init`.
3. Append an instance to `MPCoordinatorManager.controllers`.

No change to dispatch, transport, or the registry is required.

> **Notice — never block the event loop.** Handlers and lifecycle callbacks
> (`on_join` / `on_leave`) run inline on the single event loop. If a controller
> needs to do real work in response — push to mp servers, read a store, react to
> a join — it must schedule that onto a **separate task** (e.g.
> `asyncio.get_running_loop().create_task(...)`), not do it inline. Otherwise it
> head-of-line-blocks every other message on that socket, including heartbeats,
> and can cause false evictions. For CPU-bound or blocking work, use
> `run_in_executor`.

Example future controllers: `StateController` (quota reconcile + broadcast on
join via `lifecycle.on_join`), `RouteController` (blend-lookup routing across
model replicas), `KVOpsController` (pin/prefetch via `command_sender`). These
define their own model/replica/per-model-world_size schema as needed — the
backbone keeps membership pure (no model info), since one mp server may serve
several models with different parallel configs.

## Lifecycle hooks (`lifecycle.py`)

`LifecycleHooks` is a minimal `on_join`/`on_leave` callback list (not an event
bus) fired by `RegistrationController`. It lets future controllers react to
membership changes without registration importing them. **Contract:** callbacks
run inline on the event loop and must not block — heavy work (network pushes,
storage reads) must be scheduled onto a separate task so registration replies
are not delayed.

## Registry (`registry.py`)

`InstanceRegistry` maps `instance_id` → `MPInstanceNode` (address, command
socket, heartbeat timestamps, metadata). Membership is pure — no model or
parallel-config info — so a server serving multiple models is represented
correctly; model-aware indexing belongs to a future routing controller. It is
thread-safe (`threading.Lock`) and offers `stale()` for health checks. It stores
the command socket but never opens or closes it.

## Concurrency contract

- ZMQ ROUTER multiplexes all peers; concurrent registrations queue on its
  incoming buffer and are drained by the single event loop. No per-peer thread,
  no hand-rolled register buffer.
- All sockets are owned by the event loop; ZMQ sockets are never used from two
  threads.
- The registry is the only state shared with the health-check thread and is
  lock-guarded.
- The health-check thread only *detects* stale instances; eviction (which closes
  a socket) is scheduled back onto the event loop via
  `run_coroutine_threadsafe`, keeping socket lifecycle single-threaded.
- Registration is idempotent: re-registering a known instance closes the stale
  command socket and re-fires `on_join`.

## Running

```
python -m lmcache.v1.mp_coordinator
```

Configured via `LMCACHE_MP_COORDINATOR_*` environment variables — see
`MPCoordinatorConfig` in `config.py` (`PULL_URL`, `REPLY_URL`,
`HEARTBEAT_URL`, `HEARTBEAT_INTERVAL`, `INSTANCE_TIMEOUT`,
`HEALTH_CHECK_INTERVAL`).

An mp server joins by embedding a `CoordinatorClient` (`client.py`):
`start()` to register and begin heartbeats, `stop()` to deregister. Integration
into `run_cache_server` is gated on a coordinator URL being configured and is
off by default.
