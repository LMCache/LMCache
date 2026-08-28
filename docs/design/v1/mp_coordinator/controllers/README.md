# Coordinator controllers

A `Controller` is a collaborator the coordinator builds once at startup.
Subclassing the marker in
[`base.py`](../../../../../lmcache/v1/mp_coordinator/controllers/base.py) is
the whole of adding one: `build_controllers` scans this package and constructs
every class it finds.

The marker carries construction and nothing else. Consuming the cache-event
stream and holding durable state are **protocols**, matched structurally:
implement `consume` (`CacheEventConsumer`) and the fan-out subscribes you,
implement `get_durable_components` (`Durability`) and your state is routed to
an artifact. A controller
that does neither declares neither, and a class outside these packages can
still satisfy either -- which is why the ingest gate is captured without being
a view or a controller.

A setting a controller needs but the core config does not name goes in
`MPCoordinatorConfig.extra_config`, so shipping one stays a single file.

## Why discovery rather than a list

The alternative is a list in `create_app` plus a field per controller on
`CoordinatorContext`, so adding a controller means editing two files that have
nothing to do with it, and forgetting either is a silent loss -- state that is
never persisted, or a controller nobody can reach. Scanning removes both edits.

The marker matters as much as the scan: it separates a controller from the
collaborators one owns, so a helper that happens to live in this directory is
not built a second time behind its owner's back.

## Dependencies between controllers

The eviction controller reads the usage view, and its plan is only correct if
that is the same view the fleet reported into. So `ControllerRegistry` builds
**on first request**: `FleetEvictionController.from_config` asks for
`CacheUsageManager` and gets it, built if needed.

That removes construction order as a thing anyone has to reason about -- no
topological sort, no "declare your dependencies", no relying on the alphabet.
A cycle is caught and named rather than becoming a `RecursionError` deep inside
discovery.

## What discovery does not own

**Ingest order.** `create_app` registers cache-event consumers explicitly,
because the order is load-bearing: the eviction controller reads the usage view
for the batch the usage view has just consumed. Discovery returns controllers
in class-name order, which happens to be correct today and would be a trap to
depend on.

**Anything outside this package.** The ingest gate holds durable state but is
neither a view nor a controller, so `create_app` names it.
