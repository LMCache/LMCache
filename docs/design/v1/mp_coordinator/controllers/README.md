# Coordinator controllers

A `Controller` is a collaborator the coordinator builds once at startup.
Subclassing the marker in
[`base.py`](../../../../../lmcache/v1/mp_coordinator/controllers/base.py) is
the whole of adding one: `build_controllers` scans this package and constructs
every class it finds.

The marker carries **construction and lifetime**, both defaulted: override
`from_config` to read configuration or a view, `run` to do something in the
background for as long as the app serves. Every controller has a lifetime, so
these are interface members rather than an opt-in protocol -- as in
`StorageControllerInterface` under `lmcache/v1/distributed/`.

What a controller might do *besides* that is a **protocol**, matched
structurally:

| Implement | Protocol | And startup |
| --- | --- | --- |
| `consume` | `CacheEventConsumer` | subscribes you to the cache-event fan-out |
| `get_durable_components` | `Durability` | routes your state to an artifact |
| `get_routers` | `HttpRoutes` | mounts your endpoints on the coordinator |

A controller implementing none of them declares none, and a class outside
these packages can still satisfy any -- which is why the ingest gate is
captured without being a view or a controller.

A setting a controller needs but the core config does not name goes in
`MPCoordinatorConfig.extra_config`, so shipping one needs no edit here either
-- including which out-of-tree packages to load, below.

## Registering HTTP endpoints

The routers in `http_apis/` resolve their collaborator with
`ctx.controllers.get(FleetEvictionController)`, which works only for a class
the coordinator already imports. A controller that ships elsewhere instead
implements `get_routers`, returning routers built around itself:

```python
class WidgetController(Controller):
    def get_routers(self) -> tuple[APIRouter, ...]:
        router = APIRouter()

        @router.get("/widget/status")
        async def status() -> dict[str, int]:
            return {"pending": self.pending}   # closes over self

        return (router,)
```

`create_app` mounts these after the `http_apis` routers, so an in-tree path
wins a collision. Handlers bind to `self` rather than looking the owner up
per request -- that lookup is what the protocol removes.

Discovery walks subpackages, so a controller past one file can be a directory
(`controllers/widget/{controller.py, http_api.py}`) without anything outside
naming either half.

## Shipping a controller outside this tree

Nothing has to land in these directories. Name an importable package and it is
scanned the same way as this one -- `discover` takes a list of names, and the
built-in package is just the first of them.

A worked example. The package is ordinary Python, installed or just on
`PYTHONPATH`, and imports from lmcache but is never imported by it:

```
acme_controllers/
  __init__.py
  reaper/
    __init__.py
    controller.py      # the controller
    http_api.py        # the endpoints it owns
```

```python
# acme_controllers/reaper/controller.py
from contextlib import asynccontextmanager
import asyncio

from lmcache.v1.mp_coordinator.controllers.base import Controller, ControllerRuntime
from lmcache.v1.mp_coordinator.views.instance_registry import InstanceRegistry

from acme_controllers.reaper.http_api import build_router


class ReaperController(Controller):
    """Logs the fleet size on a cadence, and reports it over HTTP."""

    def __init__(self, registry: InstanceRegistry, interval: float) -> None:
        self._registry = registry
        self._interval = interval
        self.last_seen = 0

    @classmethod
    def from_config(cls, config, views) -> "ReaperController":
        # In-tree views come from the registry; settings the core config
        # does not name come from extra_config.
        return cls(
            registry=views.get(InstanceRegistry),
            interval=float(config.extra_config.get("acme.reaper_interval", 30.0)),
        )

    @asynccontextmanager
    async def run(self, runtime: ControllerRuntime):
        task = asyncio.create_task(self._sweep())
        try:
            yield
        finally:
            task.cancel()

    def get_routers(self):
        return build_router(self)

    async def _sweep(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            self.last_seen = len(self._registry.all_instances())
```

```python
# acme_controllers/reaper/http_api.py
from fastapi import APIRouter


def build_router(reaper) -> tuple[APIRouter, ...]:
    router = APIRouter()

    @router.get("/acme/reaper")
    async def status() -> dict[str, int]:
        return {"last_seen": reaper.last_seen}   # closes over the controller

    return (router,)
```

Then point the coordinator at it. One blob carries both the package and the
controller's own settings:

```bash
lmcache coordinator --extra-config '{
  "controller_packages": ["acme_controllers"],
  "acme.reaper_interval": 10
}'
```

Startup discovers `ReaperController`, builds it via `from_config`, enters its
`run`, and mounts `/acme/reaper` -- with no edit to `create_app`, the config
class, or the CLI.

It rides in `extra_config` rather than on a flag of its own, for the same
reason anything else a controller needs does: the core config should not grow
a field per out-of-tree concern. A name may address a package (walked entire)
or a single module. A name that does not import raises rather than being
skipped -- an operator asked for it, so a silent skip would look like a
controller that loaded and did nothing.

**Controllers only.** There is no `--view-package`: a view is shared fleet
state, so which views exist is the coordinator's contract. An out-of-tree
controller reads the in-tree views like any other, and whatever it needs
beyond them is its own state.

This is the model vLLM uses to find the LMCache connector: the host names an
importable path, and nothing in the host imports the plugin.

## Why discovery rather than a list

The alternative is a list in `create_app` plus a field per controller on
`CoordinatorContext`, so adding a controller means editing two files that have
nothing to do with it, and forgetting either is a silent loss -- state that is
never persisted, or a controller nobody can reach. Scanning removes both edits.

The marker matters as much as the scan: it separates a controller from the
collaborators one owns, so a helper that happens to live in this directory is
not built a second time behind its owner's back.

## What a controller may depend on

Views, and nothing else. An eviction controller reads the usage view, and its
plan is only correct if that is the same view the fleet reported into, so the
registry builds **on first request**: `FleetEvictionController.from_config`
asks for `CacheUsageManager` and gets it, built if needed. That removes
construction order as a thing anyone has to reason about -- no topological
sort, no "declare your dependencies", no relying on the alphabet.
It asks for `KeyDirectory` too, since only the node holding an L1 key can
delete it.

`from_config` cannot reach another controller: one that named a peer would
break when that peer shipped elsewhere. Shared *state* is what a view is for;
shared *policy* means the two were one controller.

## Background work

`run` is an async context manager: enter it to start background work, and the
exit half tears it down. It is handed a `ControllerRuntime` carrying only what
a controller cannot already have -- today the outbound HTTP client, which binds
to the running event loop. Everything else, including fleet membership (the
`InstanceRegistry` view), arrives through `from_config`. The runtime is a
struct rather than a bare argument so a second loop-bound collaborator can be
added without touching the signature of every controller that never asked for
one.

A context manager rather than a `start` / `stop` pair because the two failure
modes it removes are otherwise the caller's to remember: whatever a controller
started before raising is unwound for it, and one that never entered is never
torn down.

A controller raising on the way in is logged and skipped rather than fatal --
any package can supply one, and a broken controller must not cost the
coordinator the endpoints that belong to no controller, nor the controllers
that did start. Whatever the failed one does is then simply not happening,
and the log is the only notice of that.

```python
@asynccontextmanager
async def run(self, runtime: ControllerRuntime) -> AsyncIterator[None]:
    task = asyncio.create_task(self._loop(runtime.http_client))
    try:
        yield
    finally:
        task.cancel()
```

The lifespan owns only what is not any one controller's: the health-check
sweep and the checkpoint timer. It drives everything through an
`AsyncExitStack`, so registration order is teardown order reversed, and that
order is load-bearing -- timers stop before controllers so no checkpoint races
one settling, controllers before the final write so it captures what they
settled on, and the client closes last because a draining controller is still
using it.

## What discovery does not own

**Ingest order.** `create_app` registers cache-event consumers explicitly,
because the order is load-bearing: an eviction controller reads the usage view
for the batch the usage view has just consumed. Discovery returns controllers
in class-name order, which happens to be correct today and would be a trap to
depend on.

**Anything outside this package.** The ingest gate holds durable state but is
neither a view nor a controller, so `create_app` names it.

**Route conflicts.** `http_apis` mounts first, then controllers in discovery
order. A duplicate path is not detected -- FastAPI serves whichever mounted
first, so an in-tree route always beats a controller claiming the same one.
