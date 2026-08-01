# Runtime policy coordination

## Scope

This document describes the coordinator side of the runtime management-policy
API from issue #4360. The node-local API is implemented by
``multiprocess/http_apis/policy_api.py``. This phase adds a coordinator proxy
and a fleet fan-out layer; it does not migrate cache state or change the
configured storage topology.

## Endpoints

The coordinator exposes direct proxy endpoints for one registered MP server:

| Method | Path | Semantics |
| --- | --- | --- |
| ``GET`` | ``/instances/{instance_id}/config/policies`` | Read node-local state |
| ``POST`` | ``/instances/{instance_id}/config/policies/validate`` | Validate without applying |
| ``PATCH`` | ``/instances/{instance_id}/config/policies`` | Apply node-local update |

Fleet endpoints target every currently registered instance:

| Method | Path | Semantics |
| --- | --- | --- |
| ``POST`` | ``/fleet/config/policies/validate`` | Validate all targets |
| ``PATCH`` | ``/fleet/config/policies`` | Validate all, then apply all |

Fleet requests wrap the node-local update because versions are node-local:

```json
{
  "update": {
    "store_policy": "lru",
    "l1_eviction": {"tunables": {"eviction_ratio": 0.2}}
  },
  "expected_versions": {
    "node-a": 4,
    "node-b": 7
  }
}
```

The optional ``expected_versions`` map is keyed by registered instance id. A
fleet request must not put ``expected_version`` inside ``update``. For a
``PATCH`` that omits a node's version, the coordinator uses the version
returned by that node's validation response as the apply precondition. This
keeps the normal path protected against a concurrent update between validation
and apply while still allowing a caller to provide explicit versions.

## Fan-out semantics

Fleet ``PATCH`` is a two-stage best-effort operation:

1. The coordinator snapshots the registered instances and sends validation
   requests concurrently to all of them.
2. If any validation fails or a target is unreachable, no ``PATCH`` requests
   are sent. The response is ``status: "rejected"`` with one result per target.
3. If validation succeeds, the coordinator sends fenced ``PATCH`` requests
   concurrently. It reports ``status: "updated"`` only when every target
   succeeds. A failure after the barrier returns ``status: "partial"`` and
   preserves each target's status code and response body.

This is deliberately not an atomic distributed transaction. There is no
rollback in this phase: a caller can inspect the result list and retry only
the failed or stale targets. Version conflicts are reported as HTTP 409,
transport failures as HTTP 502, and validation errors as HTTP 400 when no more
specific status applies. An empty registry is HTTP 404.

Every successful node-local validation response must contain its current
``version``. A malformed response is treated as a coordinator-side validation
failure, and no apply is attempted.

## Ownership and lifecycle

``InstanceRegistry`` remains the sole source of target addresses. The policy
manager only translates an ``MPInstance`` into ``http://ip:http_port`` and
does not cache membership or policy state. The router is auto-discovered by
the coordinator's existing ``*_api.py`` convention.

The coordinator's existing lifespan-owned ``httpx.AsyncClient`` is reused for
all direct and fleet calls. Registration and heartbeat health checks can remove
an instance after the target snapshot; an in-flight request still reports that
target's result, while later calls use the new registry snapshot.

## Deliberate non-goals

- No cross-node atomic commit or rollback.
- No durable policy desired-state store.
- No automatic retry, quorum, or leader election.
- No eviction-class replacement or L2 adapter migration; those remain
  ``state_migration_required`` node-local errors until the later phase.
