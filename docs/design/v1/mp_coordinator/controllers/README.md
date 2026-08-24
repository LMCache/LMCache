# Coordinator controllers

A `Controller` is a collaborator the coordinator builds once at startup.
Subclassing the marker in
[`base.py`](../../../../../lmcache/v1/mp_coordinator/controllers/base.py) is
the whole of adding one: `build_controllers` scans this package, constructs
every class it finds, and routes whatever durable state that class advertises.

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

**Anything outside this package.** The key directory and the ingest gate hold
durable state but are not controllers, so `create_app` names them.
