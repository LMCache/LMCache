# Frontend with MP Coordinator Integration

The [frontend](../../../lmcache/lmcache_frontend/) is a standalone FastAPI
dashboard and reverse proxy over a fleet of mp cache servers. It sources fleet
membership from the [mp coordinator](../v1/mp_coordinator/README.md)'s
`InstanceRegistry` (the single source of truth) instead of discovering the
fleet itself.

Code: `lmcache/lmcache_frontend/app.py`.

## Why

The coordinator already learns the fleet by *push*: mp servers register,
heartbeat, and deregister, and a health loop evicts stale ones. The frontend
used to maintain a parallel, pull-based discovery path (a "node supplier" URL
plus a per-server `/api/nodes` fan-out) that duplicated discovery and could
drift from the coordinator. The frontend now consumes the coordinator and that
parallel path is gone.

## Design

The frontend keeps a single **flat** list of mp-server nodes (`_node_registry`);
every consumer (`/api/nodes`, the `/metrics` aggregator, the `/proxy` SSRF
guard, and the UI) reads that list. `fetch_nodes_from_coordinator` calls the
coordinator's `GET /instances` and maps each instance to one node:

```
instance {instance_id, ip, http_port}  ->  {name: "mp_<id>", host: ip, port: str(http_port)}
```

Only **membership and health** come from the coordinator. Per-server runtime
data (`/metrics`, `/version`) is not in the registry, so the frontend fetches it
directly from each node.

| Data | Source of truth | Frontend obtains it by |
| --- | --- | --- |
| Membership (who is in the fleet) | coordinator registry | `GET {coordinator}/instances` |
| Health (alive / evicted) | coordinator (heartbeat + eviction) | presence in the `/instances` list |
| Metrics / version (per-server) | the mp servers | direct fan-out to each node |

## Assumptions

- The coordinator is a **single instance** and the sole source of truth.
- Its registry is in-memory and ephemeral (rebuilt from heartbeats after a restart).
- The frontend reads from **one** `--coordinator-url` and does not aggregate across coordinators. 
- Right after a coordinator restart `GET /instances` may be temporarily empty; the frontend 
  keeps its last-known list rather than clearing it.

## What changed

- Membership comes from `--coordinator-url`, which means the `--node-supplier-url` path and
  its `/api/nodes` fan-out were removed.
- The node model is **flat**, which means one node per mp server. The previous
  proxy → children tree is gone.
- The dashboard is **read-only**: membership is owned by the coordinator, so the
  add/update/delete node endpoints and their UI were removed.
- `--config` / `--nodes` remain as a manual override and local test seed.

## Running

```
# launch the coordinator
lmcache coordinator --host 0.0.0.0 --port 9300

# launch the frontend, sourcing membership from the coordinator
python -m lmcache.lmcache_frontend.app --port 8000 --coordinator-url http://localhost:9300
```

`--coordinator-url` takes precedence over `--config` / `--nodes`. mp servers
register themselves with the coordinator via their `--coordinator-*` flags (see
the [mp coordinator](../v1/mp_coordinator/README.md) doc).

## Testing

`tests/lmcache_frontend/test_coordinator.py` mocks `httpx` to cover the mapping
(populated / empty fleet, unreachable → `[]`) and the frontend endpoints
(`/api/nodes` reflects membership, the SSRF guard rejects unregistered hosts).
No coordinator or mp server process is required.
