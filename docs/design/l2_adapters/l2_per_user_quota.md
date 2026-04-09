# L2 Per-User Quota Design

This document describes the per-user quota mechanism for L2 adapters: how
per-user storage limits are enforced, how user identity propagates through the
system, and what changes are needed across the codebase.

## Motivation

Currently, L2 eviction operates on **aggregate** storage usage. A single
`trigger_watermark` governs whether eviction fires, and the LRU policy evicts
globally without regard to who stored what. In a multi-tenant serving
environment (multiple users sharing a single vLLM + LMCache deployment), one
user's burst of traffic can fill L2 and push out other users' cached KV data.

Per-user quotas add a second eviction dimension: **each user has an independent
storage budget**. When a user exceeds their budget, only that user's
least-recently-used keys are evicted — other users' cached data remains
untouched.

## Design Overview

```
vLLM API Server
  │  sends user_id directly on IPCCacheEngineKey
  ▼
LMCache MP Server
  │  reads key.user_id
  │  ipc_key_to_object_keys(..., user_id=key.user_id)
  ▼
ObjectKey(chunk_hash, model_name, kv_rank, user_id="alice")
  │                                         ▲
  │                            user_id IS part of key identity
  │                            (participates in __eq__ / __hash__)
  │                            same tokens + different user = different key
  ▼
L1 Manager → StoreController → L2 Adapter
  │                                │
  │                    reads key.user_id to
  │                    track per-user bytes
  ▼
_notify_keys_stored(keys)  ──►  L2EvictionPolicy bridge
                                     │
                                     ▼
                              UserLRUEvictionPolicy
                              ┌──────────────────────┐
                              │ "alice" → OrderedDict │
                              │ "bob"   → OrderedDict │
                              │ ...                    │
                              └──────────────────────┘

L2EvictionController (every 1s):
  for each adapter state:
    per_user_usage = adapter.get_per_user_usage()
    for user_id, usage in per_user_usage:
      if usage > watermark * quota(user_id):
        actions = policy.get_eviction_actions(ratio, user_id=user_id)
        adapter.delete(actions.keys)   ← only that user's keys
```

## Key Design Decisions

### 1. Strict User Isolation — `user_id` in ObjectKey Identity

User identity is a **full identity field** on `ObjectKey`, participating in
`__eq__` and `__hash__`:

```python
@dataclass(frozen=True)
class ObjectKey:
    chunk_hash: bytes
    model_name: str
    kv_rank: int
    user_id: str = ""
```

Two ObjectKeys with the same (chunk_hash, model_name, kv_rank) but different
user_ids are **different keys**. If Alice and Bob send identical token
sequences, they produce separate ObjectKeys and store separate copies in both
L1 and L2.

**Why strict isolation?**

- **No cross-user interference:** Evicting Alice's keys never affects Bob's
  cache hits. Each user's cached data is fully independent.
- **Simple ownership:** Every key belongs to exactly one user — no ambiguity,
  no "first-storer-wins" races, no shared-ownership accounting.
- **Clean retrieval:** Lookup and retrieve naturally scope to the correct
  user because `user_id` is part of the key. No special filtering needed.
- **Predictable quotas:** Per-user byte accounting is exact. A user's usage
  equals the sum of their keys' sizes, with no shared entries to split.

**Trade-off — storage duplication:** The same token sequence cached by N
users consumes N times the storage (in both L1 and L2). This is the cost of
strict isolation. In practice, this is acceptable for multi-tenant deployments
where isolation guarantees outweigh storage efficiency, and where distinct
users typically have distinct prompts anyway.

**Backward compatibility:** `user_id` defaults to `""`. Legacy deployments
(no user_id in request_id) produce keys with `user_id=""` that all share the
same namespace — identical to today's behavior.

### 2. User ID Propagation: API → vLLM → LMCache

User identity flows through the existing `extra_args` / `kv_transfer_params`
mechanism that LMCache already uses for per-request config (e.g.,
`lmcache.skip_save`). No vLLM core changes are needed.


**API caller** includes `user_id` in the request body:

```json
{
  "model": "llama-3-8b",
  "messages": [...],
  "extra_args": {
    "kv_transfer_params": {
      "lmcache.user_id": "alice"
    }
  }
}
```

**Propagation path:**

The scheduler adapter has `user_id` (from `request_configs`), but the
worker adapter does not — vLLM's connector framework passes only
`request_id` + `LoadStoreOp` to the worker, and LMCache cannot modify
what vLLM dispatches. The solution is **server-side session caching**:
the scheduler sends `user_id` during LOOKUP, the server stores it in
the session, and applies it when the worker's STORE/RETRIEVE arrives.

```
API request body
  │  extra_args.kv_transfer_params.lmcache.user_id = "alice"
  ▼
vLLM SamplingParams.extra_args  (vLLM passes this through unchanged)
  ▼
extract_request_configs()       (vllm_v1_adapter.py — already extracts lmcache.* keys)
  → request_configs = {"lmcache.user_id": "alice"}
  ▼
LMCacheConnectorV1              (reads user_id from request_configs)
  → user_id = request_configs.get("lmcache.user_id", "")
  ▼
LMCacheMPSchedulerAdapter       (only the scheduler has user_id)
  → _create_key(..., user_id=user_id)
  ▼
IPCCacheEngineKey(user_id="alice", ...)  ──LOOKUP──►  MP Server
                                                       │ session.user_id = "alice"
                                                       │ (cached for this request_id)
                                                       ▼
LMCacheMPWorkerAdapter          (no user_id available)
  → _create_key(...)  ← user_id="" on the key
  ▼
IPCCacheEngineKey(user_id="", ...) ──STORE──►  MP Server
                                                │ session = get(key.request_id)
                                                │ user_id = session.user_id → "alice"
                                                ▼
                                    ipc_key_to_object_keys(..., user_id="alice")
                                                ▼
                                    ObjectKey(user_id="alice", ...)
```

The protocol guarantees LOOKUP always precedes STORE/RETRIEVE for a given
request, so `user_id` is always available in the session when the worker's
request arrives.

`user_id` is added as an identity field on `IPCCacheEngineKey` (with
`compare=True`, unlike `request_id`). `request_id` is ephemeral session
metadata — two requests with different `request_id`s but the same tokens
should hit the same cache. `user_id` is the opposite: same tokens from
different users must **not** match. This is consistent with `user_id`
participating in `ObjectKey.__eq__` / `__hash__`.

```python
@dataclass(order=True, frozen=True)
class IPCCacheEngineKey:
    model_name: str
    world_size: int
    worker_id: int | None
    token_ids: tuple[int, ...]
    start: int
    end: int
    request_id: str = field(compare=False)   # position 7 — unchanged
    user_id: str = ""                         # position 8 — appended at end
```

`user_id` is placed **after** `request_id` (at the end) to preserve
msgspec wire compatibility. `IPCCacheEngineKey` is serialized positionally
via `msgspec.msgpack`; appending `user_id` at the end means an old client
sending 7 fields to a new server will decode correctly with `user_id`
defaulting to `""`. No changes to existing field positions.

`no_worker_id_version()` must be updated to preserve `user_id` when
copying the key (currently reconstructs explicitly and would drop the
new field).

When `user_id` is empty (default), per-user quota logic is skipped and
behavior is identical to today.

### 3. Dynamic Per-User Quotas via `QuotaManager`

Per-user quotas are **dynamic** — they can be created, updated, and deleted
at runtime via the HTTP API. A `QuotaManager` holds the per-user limit
registry and is queried by the eviction controller each cycle.

**Quota lookup rules:**

1. If a user_id has an explicit entry in the quota registry → use that limit.
2. If a user_id has **no** entry → effective limit is **0 bytes**.
   - Stores are still allowed (we never reject writes on the hot path).
   - At the next eviction cycle (~1s), the controller sees
     `usage > 0 > limit 0` and triggers eviction.
   - The user's keys are evicted, freeing the space.

This means the system is **allowlist-based**: only users with an explicit
quota can retain cached data. Unknown users get temporary write access, but
their data is cleaned up within one eviction cycle.

Per-user quotas are enabled by choosing the `UserLRU` eviction policy.
If the operator does not want per-user quotas, they simply use the `LRU`
policy — no special "disable" flag is needed.

### 4. HTTP API for Quota Management

The existing FastAPI HTTP server (`lmcache/v1/multiprocess/http_server.py`)
already serves `/api/healthcheck`, `/api/status`, and `/api/clear-cache`.
Add quota management endpoints:

```
PUT    /api/quota/{user_id}          Set/update quota for a user
GET    /api/quota/{user_id}          Get quota and current usage for a user
DELETE /api/quota/{user_id}          Remove quota (user's data evicted next cycle)
GET    /api/quota                    List all quotas and per-user usage
```

**`PUT /api/quota/{user_id}`** — Set or update a user's quota.
`limit_gb` is required.

```json
// Request body
{"limit_gb": 2.0}

// Response
{"user_id": "alice", "limit_gb": 2.0, "status": "ok"}
```

**`GET /api/quota/{user_id}`** — Get quota and current usage.

```json
// Response
{
  "user_id": "alice",
  "limit_gb": 2.0,
  "current_usage_gb": 1.3,
  "exists": true
}
```

**`DELETE /api/quota/{user_id}`** — Remove quota entry. The user's cached
data will be evicted at the next eviction cycle (effective limit becomes 0).

```json
// Response
{"user_id": "alice", "status": "removed"}
```

**`GET /api/quota`** — List all registered quotas with per-user usage.

```json
// Response
{
  "users": {
    "alice": {"limit_gb": 2.0, "current_usage_gb": 1.3},
    "bob":   {"limit_gb": 5.0, "current_usage_gb": 4.1}
  }
}
```

### 5. Per-Adapter Eviction Config

The `UserLRU` policy replaces `LRU` when per-user quotas are desired.
No per-user limit field in the static config — quotas are managed
entirely at runtime via the HTTP API.

```json
{
  "eviction": {
    "eviction_policy": "UserLRU",
    "trigger_watermark": 0.8,
    "eviction_ratio": 0.2
  }
}
```

When `eviction_policy` is `"LRU"`, per-user quota logic is not applied —
existing behavior is unchanged.

## Component Changes

### 1. `ObjectKey` — Add `user_id` identity field

**File:** `lmcache/v1/distributed/api.py`

```python
@dataclass(frozen=True)
class ObjectKey:
    chunk_hash: bytes
    model_name: str
    kv_rank: int
    user_id: str = ""
```

`user_id` participates in `__eq__` and `__hash__` (default behavior for
frozen dataclass fields). Same content from different users produces
different ObjectKeys.

Update `ipc_key_to_object_keys()` to accept and forward `user_id`:

```python
def ipc_key_to_object_keys(
    ipc_key: IPCCacheEngineKey,
    chunk_hashes: list[bytes],
    user_id: str = "",
) -> list[ObjectKey]:
    ...
    storage_keys.append(
        ObjectKey(
            chunk_hash=chunk_hash,
            model_name=ipc_key.model_name,
            kv_rank=kv_rank,
            user_id=user_id,
        )
    )
```

### 2. Server — Session-based `user_id` resolution

**File:** `lmcache/v1/multiprocess/server.py`

The server resolves `user_id` for each request using two sources:

1. **From the IPC key** — if `key.user_id` is set (scheduler-side LOOKUP).
2. **From the session** — if `key.user_id` is empty (worker-side
   STORE/RETRIEVE), fall back to the session's cached `user_id`.

In `MPCacheEngine.lookup()` — store user_id on the session:

```python
session = self.session_manager.get_or_create(key.request_id)
if key.user_id:
    session.user_id = key.user_id
user_id = session.user_id
obj_keys = ipc_key_to_object_keys(key, chunk_hashes, user_id=user_id)
```

In `MPCacheEngine.store()` and `MPCacheEngine.retrieve()` — read from
session (worker's key has `user_id=""`):

```python
session = self.session_manager.get_or_create(key.request_id)
user_id = key.user_id or session.user_id
obj_keys = ipc_key_to_object_keys(key, chunk_hashes, user_id=user_id)
```

**File:** `lmcache/v1/multiprocess/session.py`

Add `user_id: str = ""` field to the `Session` class.

### 3. vLLM Adapter Layer — Scheduler-side only

Only the **scheduler adapter** needs changes. The worker adapter is
untouched — user_id resolution happens on the server via sessions.

**File:** `lmcache/integration/vllm/vllm_v1_adapter.py`

In `LMCacheConnectorV1.get_num_new_matched_tokens()`, extract `user_id`
from the already-parsed `request_configs` and pass it when calling the
lookup client:

```python
request_configs = extract_request_configs(request.sampling_params)
user_id = (request_configs or {}).get("lmcache.user_id", "")

num_external_hit_tokens = self.lookup_client.lookup(
    token_ids,
    lookup_id=req_id,
    request_configs=request_configs,
    user_id=user_id,
)
```

**File:** `lmcache/integration/vllm/vllm_multi_process_adapter.py`

Only the **scheduler adapter** changes:

- `LMCacheMPSchedulerAdapter.maybe_submit_lookup_request(request_id, token_ids, user_id="")`
- `LMCacheMPSchedulerAdapter.free_lookup_locks(..., user_id="")`

Scheduler adapter's `_create_key()` passes `user_id` through:

```python
def _create_key(self, token_ids, start, end, request_id, user_id=""):
    return IPCCacheEngineKey(
        model_name=self.model_name,
        world_size=self.world_size,
        worker_id=None,
        token_ids=tuple(token_ids),
        start=start, end=end,
        user_id=user_id,
        request_id=request_id,
    )
```

**`LMCacheMPWorkerAdapter` is unchanged.** Its `_create_key()` produces
keys with `user_id=""`. The server resolves the actual user_id from the
session (populated earlier during the scheduler's LOOKUP).

### 4. L2 Adapter Interface — Add `get_per_user_usage()`

**File:** `lmcache/v1/distributed/l2_adapters/base.py`

Add a new method to `L2AdapterInterface`:

```python
def get_per_user_usage(self) -> dict[str, tuple[float, float]]:
    """Return per-user L2 storage utilization.

    Returns:
        dict[str, tuple[float, float]]: Mapping of user_id to
            (current_usage, usage_after_ongoing_eviction) where
            each value is bytes used (NOT a fraction). The caller
            compares against the configured per-user limit.

    The default returns an empty dict (no per-user tracking).
    """
    return {}
```

**Why return bytes, not fractions?** Per-user limits are absolute (e.g., 2GB),
not fractions of total capacity. The controller compares
`per_user_bytes > per_user_limit_bytes` directly.

### 4. Adapter Implementations — Track per-user bytes

Each adapter that supports eviction needs to maintain:
- `_per_user_size_bytes: dict[str, int]` — bytes stored per user

No separate `_key_to_user` mapping is needed because `user_id` is part of
ObjectKey identity — `key.user_id` is always authoritative. The adapter reads
it directly from the key.

**File:** `lmcache/v1/distributed/l2_adapters/mock_l2_adapter.py`

In `_execute_store_in_the_loop`, when a key is stored:
```python
user_id = key.user_id
if user_id:
    self._per_user_size_bytes[user_id] = (
        self._per_user_size_bytes.get(user_id, 0) + obj_size
    )
```

In `delete()`, when a key is removed:
```python
user_id = key.user_id
if user_id:
    self._per_user_size_bytes[user_id] = (
        self._per_user_size_bytes.get(user_id, 0) - obj.get_size()
    )
```

Implement `get_per_user_usage()`:
```python
def get_per_user_usage(self) -> dict[str, tuple[float, float]]:
    with self._lock:
        return {
            uid: (bytes_used, bytes_used)
            for uid, bytes_used in self._per_user_size_bytes.items()
            if bytes_used > 0
        }
```

**File:** `lmcache/v1/distributed/l2_adapters/nixl_store_l2_adapter.py`
**File:** `lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py`

Same pattern. The NixlStore adapter already tracks pool-based sizes; add a
parallel per-user counter. The NativeConnector adapter already does client-side
size tracking; extend it with per-user accounting.

### 5. `UserLRUEvictionPolicy` — Per-user LRU tracking

**File (new):** `lmcache/v1/distributed/eviction_policy/user_lru.py`

`get_eviction_actions` gains an optional `user_id` parameter. When set,
eviction is scoped to that user's LRU list. When `None` (default),
eviction is global across all users — backward compatible with the
existing LRU interface.

```python
class UserLRUEvictionPolicy(EvictionPolicy):

    def __init__(
        self,
        default_destination: EvictionDestination = EvictionDestination.DISCARD,
    ):
        self._lock = threading.Lock()
        self._per_user_order: dict[str, OrderedDict[ObjectKey, None]] = {}
        self._default_destination = default_destination

    def on_keys_created(self, keys: list[ObjectKey]):
        with self._lock:
            for key in reversed(keys):
                user_id = key.user_id
                if user_id not in self._per_user_order:
                    self._per_user_order[user_id] = OrderedDict()
                user_order = self._per_user_order[user_id]
                if key in user_order:
                    user_order.move_to_end(key)
                else:
                    user_order[key] = None

    def on_keys_touched(self, keys: list[ObjectKey]):
        with self._lock:
            for key in reversed(keys):
                user_id = key.user_id
                user_order = self._per_user_order.get(user_id)
                if user_order and key in user_order:
                    user_order.move_to_end(key)

    def on_keys_removed(self, keys: list[ObjectKey]):
        with self._lock:
            for key in keys:
                user_id = key.user_id
                user_order = self._per_user_order.get(user_id)
                if user_order:
                    user_order.pop(key, None)
                    if not user_order:
                        del self._per_user_order[user_id]

    def get_eviction_actions(
        self,
        expected_ratio: float,
        user_id: str | None = None,
    ) -> list[EvictionAction]:
        """Select victims, optionally scoped to a user.

        Args:
            expected_ratio: Fraction of keys to evict.
            user_id: If set, evict from this user's list only.
                If None, evict globally across all users.
        """
        with self._lock:
            if user_id is not None:
                order = self._per_user_order.get(user_id)
                if not order:
                    return []
                pool = list(order.keys())
            else:
                pool = []
                for user_order in self._per_user_order.values():
                    pool.extend(user_order.keys())

            if not pool:
                return []

            expected_ratio = max(0.0, min(1.0, expected_ratio))
            target = int(len(pool) * expected_ratio)
            if expected_ratio > 0 and target == 0 and len(pool) > 0:
                target = 1
            if target == 0:
                return []

            return [EvictionAction(
                keys=pool[:target],
                destination=self._default_destination,
            )]
```

**`EvictionPolicy` abstract class** — add `user_id: str | None = None`
to `get_eviction_actions`. Existing implementations (`LRUEvictionPolicy`,
`NoOpEvictionPolicy`) accept and ignore it — backward compatible.

### 7. `QuotaManager` — Dynamic per-user quota registry

**File (new):** `lmcache/v1/distributed/quota_manager.py`

```python
class QuotaManager:
    """Thread-safe registry of per-user storage quotas.

    Queried by the eviction controller each cycle. Updated at runtime
    via HTTP API. Users not in the registry have an effective quota of
    0 bytes (their data is evicted at the next cycle).

    Enforces a capacity invariant: sum of all quotas ≤ total adapter
    capacity. set_quota() rejects requests that would violate this.
    """

    def __init__(self, total_capacity_bytes: int):
        self._lock = threading.Lock()
        self._quotas: dict[str, int] = {}  # user_id -> limit in bytes
        self._total_capacity_bytes = total_capacity_bytes

    def get_limit_bytes(self, user_id: str) -> int:
        """Return the quota for a user. 0 if not registered."""
        with self._lock:
            return self._quotas.get(user_id, 0)

    def set_quota(self, user_id: str, limit_gb: float) -> None:
        """Set quota for a user. limit_gb is required.

        Raises:
            ValueError: If adding/updating this quota would cause the
                sum of all quotas to exceed total adapter capacity.
        """
        new_limit = int(limit_gb * (1024 ** 3))
        with self._lock:
            old_limit = self._quotas.get(user_id, 0)
            current_total = sum(self._quotas.values())
            new_total = current_total - old_limit + new_limit
            if new_total > self._total_capacity_bytes:
                raise ValueError(
                    f"Cannot set quota for {user_id}: sum of quotas "
                    f"({new_total / (1024**3):.2f} GB) would exceed "
                    f"adapter capacity "
                    f"({self._total_capacity_bytes / (1024**3):.2f} GB)"
                )
            self._quotas[user_id] = new_limit

    def remove_quota(self, user_id: str):
        """Remove quota. User's data will be evicted next cycle."""
        with self._lock:
            self._quotas.pop(user_id, None)

    def get_all_quotas(self) -> dict[str, int]:
        """Return a snapshot of all quotas (user_id -> bytes)."""
        with self._lock:
            return dict(self._quotas)
```

The capacity check in `set_quota()` prevents over-provisioning: the sum of
all per-user quotas can never exceed the adapter's total capacity. This
means that if every user fills their quota simultaneously, the aggregate
usage stays within bounds.

Note that the global watermark (e.g., 0.8) may still trigger before any
individual user exceeds their quota — this is expected. The watermark is a
safety margin for overall capacity management and works independently of
per-user quotas.

The `QuotaManager` is created by `StorageManager` (initialized with the
adapter's `max_capacity_bytes`) and shared with both the
`L2EvictionController` and the HTTP server (via `app.state`).

### 8. Eviction Config — Add `"UserLRU"` policy option

**File:** `lmcache/v1/distributed/config.py`

```python
@dataclass
class EvictionConfig:
    eviction_policy: Literal["LRU", "UserLRU", "noop"]
    trigger_watermark: float = 0.8
    eviction_ratio: float = 0.2
```

No per-user limit in the static config. Quotas are managed entirely at
runtime via the `QuotaManager` and HTTP API.

**File:** `lmcache/v1/distributed/l2_adapters/config.py`

Add `"UserLRU"` to the allowed values in `_parse_eviction_config()`.

### 9. L2 Eviction Controller — Per-user eviction trigger

**File:** `lmcache/v1/distributed/storage_controllers/eviction_controller.py`

The controller receives a reference to the `QuotaManager`. Each eviction
cycle it finds **all** users who violate their watermark threshold and
evicts from each of them. After one cycle, no user should be violating.

```python
class L2EvictionController(StorageControllerInterface):
    def __init__(
        self,
        l2_adapter_states: list[L2AdapterEvictionState],
        quota_manager: QuotaManager,
    ):
        self._adapter_states = l2_adapter_states
        self._quota_manager = quota_manager
        ...

    def _check_and_evict(self, state: L2AdapterEvictionState):
        watermark = state.eviction_config.trigger_watermark
        eviction_ratio = state.eviction_config.eviction_ratio
        policy = state.eviction_policy

        if isinstance(policy, UserLRUEvictionPolicy):
            # --- UserLRU: per-user watermark check ---
            per_user_usage = state.adapter.get_per_user_usage()
            for user_id, (user_bytes, _) in per_user_usage.items():
                limit = self._quota_manager.get_limit_bytes(user_id)
                if user_bytes <= watermark * limit:
                    continue

                logger.info(
                    "User %s L2 usage %.2f GB exceeds watermark "
                    "(%.0f%%) of quota %.2f GB; evicting.",
                    user_id,
                    user_bytes / (1024 ** 3),
                    watermark * 100,
                    limit / (1024 ** 3),
                )
                actions = policy.get_eviction_actions(
                    eviction_ratio, user_id=user_id
                )
                for action in actions:
                    self._execute_eviction_action(state.adapter, action)
        else:
            # --- LRU/NoOp: global aggregate watermark (unchanged) ---
            current_usage, _ = state.adapter.get_usage()
            if current_usage < 0 or current_usage < watermark:
                return
            logger.info(
                "L2 usage %.2f above watermark %.2f; triggering eviction.",
                current_usage, watermark,
            )
            actions = policy.get_eviction_actions(eviction_ratio)
            for action in actions:
                self._execute_eviction_action(state.adapter, action)
```

The controller branches on policy type:
- **`UserLRU`**: iterates every user, checks `usage > watermark * quota`,
  evicts from each violator's LRU list. Users within quota are untouched.
- **`LRU` / `NoOp`**: existing global aggregate watermark check, unchanged.

**Single watermark, uniformly applied:** The same `trigger_watermark` (e.g.,
0.8) is applied to each user's quota. Because the capacity check enforces
`sum(quotas) ≤ capacity`, the aggregate of **registered** users can never
reach `watermark * capacity` without an individual user first exceeding
`watermark * their_quota`. This makes a global aggregate check redundant
for registered users — it is removed.

**Unknown user behavior:** When a user has no entry in the quota manager,
`get_limit_bytes()` returns 0. `watermark * 0 = 0`, so any non-zero usage
triggers eviction on the next cycle. The write itself is never rejected —
the data lives temporarily in L1/L2 until the eviction loop cleans it up.

**Edge case — unregistered user burst:** Unregistered users' temporary data
(quota=0, evicted next cycle) is outside the `sum(quotas) ≤ capacity`
invariant. In theory, many concurrent unregistered users could collectively
push aggregate usage above capacity during the ~1s eviction window. In
practice this is bounded: adapters reject stores when physically full
(e.g., mock adapter skips keys when `current_size + obj_size > max_capacity`).
The temporary data is cleaned up within one eviction cycle.

### 9. Factory — Register UserLRU

**File:** `lmcache/v1/distributed/eviction_policy/factory.py`

```python
def CreateEvictionPolicy(eviction_config: EvictionConfig) -> EvictionPolicy:
    if eviction_config.eviction_policy == "LRU":
        return LRUEvictionPolicy()
    elif eviction_config.eviction_policy == "UserLRU":
        return UserLRUEvictionPolicy()
    elif eviction_config.eviction_policy == "noop":
        return NoOpEvictionPolicy()
    ...
```

**File:** `lmcache/v1/distributed/eviction_policy/__init__.py`

Add `UserLRUEvictionPolicy` to exports.

## Data Flow: Per-User Eviction Cycle

```
[Background thread — every 1s]
  │
  ▼
for each L2AdapterEvictionState:
  │
  for each user in adapter.get_per_user_usage():
  │
  ├─ alice: 1.7 GB > 0.8 * 2.0 GB = 1.6 GB?  → yes
  │    policy.get_eviction_actions(0.2, user_id="alice")
  │      → picks 20% of alice's LRU keys
  │    adapter.delete(keys)
  │      → removes keys, fires _notify_keys_deleted
  │        → policy.on_keys_removed, adapter decrements alice's bytes
  │
  ├─ bob: 0.8 GB > 0.8 * 2.0 GB = 1.6 GB?  → no, skip
  │
  └─ eve (unregistered, limit=0): 0.5 GB > 0.8 * 0 = 0?  → yes
       policy.get_eviction_actions(0.2, user_id="eve")
         → picks 20% of eve's LRU keys
       adapter.delete(keys)
```

## Data Flow: Store (Key Created with User Scope)

```
vLLM sends STORE with IPCCacheEngineKey(user_id="alice", ...)
  │
  ▼
MPCacheEngine.store(key, ...)
  │ obj_keys = ipc_key_to_object_keys(key, hashes, user_id=key.user_id)
  │            → [ObjectKey(hash, "llama-3-8b", rank, user_id="alice"), ...]
  ▼
StorageManager.reserve_write(obj_keys, ...)
  │  keys carry user_id="alice" through L1
  │  NOTE: if bob also stores the same tokens, his keys have
  │  user_id="bob" and are DIFFERENT keys — both are stored.
  ▼
L1Manager.finish_write(keys)
  │  fires on_l1_keys_write_finished(keys)  — keys have user_id
  ▼
StoreListener → StoreController._process_new_keys(keys)
  │  calls adapter.submit_store_task(keys, objects)  — keys have user_id
  ▼
MockL2Adapter._execute_store_in_the_loop(keys, objects, task_id)
  │  for key in keys:
  │    self._per_user_size_bytes[key.user_id] += obj_size
  │  fires _notify_keys_stored(stored_keys)  — keys have user_id
  ▼
L2EvictionPolicy bridge → UserLRUEvictionPolicy.on_keys_created(keys)
  │  for key in keys:
  │    user_id = key.user_id  → "alice"
  │    self._per_user_order["alice"][key] = None
```

## Data Flow: Lookup / Retrieve (User-Scoped)

```
vLLM sends LOOKUP with IPCCacheEngineKey(user_id="alice", ...)
  │
  ▼
MPCacheEngine.lookup(key, ...)
  │ obj_keys = ipc_key_to_object_keys(key, hashes, user_id=key.user_id)
  │            → ObjectKeys with user_id="alice"
  │
  │ L2 adapter lookup matches ONLY alice's keys (because user_id
  │ participates in __eq__). Bob's keys with the same token hash
  │ are different ObjectKeys and will NOT match.
  ▼
RETRIEVE uses the same user-scoped ObjectKeys
  │ adapter.submit_load_task(keys, objects)  — keys have user_id="alice"
  │ adapter fires _notify_keys_accessed(keys)
  ▼
UserLRUEvictionPolicy.on_keys_touched(keys)
  │ moves keys to end of alice's LRU list
```

## Configuration

### Example: L2 adapter with per-user quota

```json
{
  "type": "mock",
  "max_size_gb": 10,
  "mock_bandwidth_gb": 4,
  "eviction": {
    "eviction_policy": "UserLRU",
    "trigger_watermark": 0.8,
    "eviction_ratio": 0.2
  }
}
```

| Field               | Type    | Default | Description                                           |
|---------------------|---------|---------|-------------------------------------------------------|
| `eviction_policy`   | string  | —       | `"LRU"`, `"UserLRU"`, or `"noop"`. Required.         |
| `trigger_watermark` | float   | `0.8`   | Usage fraction to trigger eviction. Applied uniformly: for `LRU`, against aggregate capacity; for `UserLRU`, against each user's quota. |
| `eviction_ratio`    | float   | `0.2`   | Fraction of keys to evict each cycle.                 |

Per-user limits are not in the static config — they are managed at runtime
via the HTTP API (`PUT /api/quota/{user_id}`). Choosing `UserLRU` enables
the per-user quota machinery; choosing `LRU` disables it entirely.

### Runtime quota management

Quotas are managed entirely at runtime via the HTTP API:

```bash
# Grant alice 2 GB quota
curl -X PUT http://localhost:8000/api/quota/alice -d '{"limit_gb": 2.0}'

# Check alice's usage
curl http://localhost:8000/api/quota/alice

# Revoke alice's quota (data evicted next cycle)
curl -X DELETE http://localhost:8000/api/quota/alice

# List all quotas
curl http://localhost:8000/api/quota
```

### Single watermark, two modes

The `trigger_watermark` is a single knob with uniform semantics:

| Policy    | Trigger condition | Description |
|-----------|-------------------|-------------|
| `LRU`     | `aggregate_usage ≥ watermark * total_capacity` | Standard global eviction (unchanged). |
| `UserLRU` | `user_usage > watermark * user_quota` for any user | Per-user eviction. No separate global check needed — the capacity invariant (`sum(quotas) ≤ capacity`) guarantees the aggregate stays within bounds. |

With `UserLRU`, there is **no global aggregate check**. It is mathematically
redundant: if no individual user exceeds `watermark * their_quota`, the
aggregate cannot exceed `watermark * capacity` (because
`sum(quotas) ≤ capacity`). This eliminates the confusing case where the
global watermark fires before any user is over their individual limit.

## Backward Compatibility

- **No user_id from vLLM:** `IPCCacheEngineKey.user_id` defaults to `""`.
  All keys have `user_id=""` — they all share the same (empty-user)
  namespace, exactly like today's behavior. Per-user quota logic is skipped
  because `get_per_user_usage()` returns empty or only has the `""` entry.
- **`eviction_policy: "LRU"`:** Per-user quota logic is not active. The
  watermark is applied against aggregate capacity as before. Existing
  behavior is fully unchanged.
- **ObjectKey equality change:** Adding `user_id` to ObjectKey identity IS a
  behavioral change, but since `user_id` defaults to `""`, all existing keys
  (with no user_id) remain equal to each other. Only when user_id is
  actively set do keys diverge. Existing tests that construct
  `ObjectKey(hash, model, rank)` continue to work — the 3-arg form uses
  `user_id=""` by default.
- **Serialization — what if an adapter doesn't update?** Each adapter uses
  ObjectKey differently as a storage key:

  | Adapter | How ObjectKey is used as storage key | Impact |
  |---------|-------------------------------------|--------|
  | `MockL2Adapter` | Python dict key (`dict[ObjectKey, ...]`) | **No change needed.** `__hash__` includes `user_id` automatically. With `user_id=""` (LRU mode), hashes are unchanged from today. |
  | `NixlStoreL2Adapter` | Python dict key (`dict[ObjectKey, ...]`) | **No change needed.** Same as mock. |
  | `NativeConnectorL2Adapter` | Explicit string serialization via `_object_key_to_string()`: `"{model}@{kv_rank}@{hash}"` | **Must update** to include `user_id` for UserLRU. Without the update, different users' keys serialize to the same string → storage collision. |

  **With regular `LRU` policy (no user_id set):** All keys have `user_id=""`.
  Even if `_object_key_to_string()` is not updated, there are no collisions
  because all keys share the same empty user_id. **Adapters work unchanged.**

  **With `UserLRU` policy (user_id set):** Adapters with explicit string
  serialization (currently only `NativeConnectorL2Adapter`) must include
  `user_id` in the serialized form, e.g.:
  ```python
  def _object_key_to_string(key: ObjectKey) -> str:
      if key.user_id:
          return f"{key.user_id}@{key.model_name}@{key.kv_rank:08x}@{key.chunk_hash.hex()}"
      return f"{key.model_name}@{key.kv_rank:08x}@{key.chunk_hash.hex()}"
  ```
  The empty-user_id branch preserves the existing format for backward
  compatibility with data already stored in Redis/FS.

- **Listener interface:** `L2AdapterListener` method signatures are unchanged.
  User_id flows through `ObjectKey.user_id`, not through callback parameters.

## File Change Summary

### New Files

| File | Description |
|------|-------------|
| `lmcache/v1/distributed/eviction_policy/user_lru.py` | `UserLRUEvictionPolicy` — per-user LRU tracking |
| `lmcache/v1/distributed/quota_manager.py` | `QuotaManager` — thread-safe per-user quota registry |

### Modified Files

| File | Change |
|------|--------|
| `lmcache/v1/distributed/api.py` | Add `user_id: str = ""` to `ObjectKey` (participates in eq/hash); add `user_id` param to `ipc_key_to_object_keys()` |
| `lmcache/v1/multiprocess/custom_types.py` | Append `user_id: str = ""` to `IPCCacheEngineKey` (after `request_id`, preserves wire compat); update `no_worker_id_version()` to preserve `user_id` |
| `lmcache/v1/multiprocess/server.py` | Resolve `user_id` from IPC key or session; pass to `ipc_key_to_object_keys()` in `store()`, `retrieve()`, `lookup()` |
| `lmcache/v1/multiprocess/session.py` | Add `user_id: str = ""` field to `Session` class |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | Extract `lmcache.user_id` from `request_configs`; pass `user_id` to scheduler lookup calls |
| `lmcache/integration/vllm/vllm_multi_process_adapter.py` | Add `user_id=""` param to scheduler adapter's `maybe_submit_lookup_request`, `free_lookup_locks`, `_create_key`. Worker adapter unchanged. |
| `lmcache/v1/distributed/eviction.py` | Add `user_id: str \| None = None` param to `EvictionPolicy.get_eviction_actions()` |
| `lmcache/v1/distributed/eviction_policy/lru.py` | Accept (and ignore) `user_id` param in `get_eviction_actions()` |
| `lmcache/v1/distributed/eviction_policy/noop.py` | Accept (and ignore) `user_id` param in `get_eviction_actions()` |
| `lmcache/v1/distributed/eviction_policy/factory.py` | Register `"UserLRU"` → `UserLRUEvictionPolicy` |
| `lmcache/v1/distributed/eviction_policy/__init__.py` | Export `UserLRUEvictionPolicy` |
| `lmcache/v1/distributed/config.py` | Add `"UserLRU"` to `eviction_policy` literal |
| `lmcache/v1/distributed/l2_adapters/config.py` | Add `"UserLRU"` to allowed values in `_parse_eviction_config()` |
| `lmcache/v1/distributed/l2_adapters/base.py` | Add `get_per_user_usage() -> dict[str, tuple[float, float]]` default method |
| `lmcache/v1/distributed/l2_adapters/mock_l2_adapter.py` | Add `_per_user_size_bytes` tracking; implement `get_per_user_usage()`; update store/delete to maintain per-user byte counters |
| `lmcache/v1/distributed/l2_adapters/nixl_store_l2_adapter.py` | Same per-user tracking pattern as mock adapter |
| `lmcache/v1/distributed/l2_adapters/native_connector_l2_adapter.py` | Same per-user tracking; update `_object_key_to_string()` to include `user_id` |
| `lmcache/v1/distributed/l2_adapters/fs_l2_adapter.py` | Update `_object_key_to_filename()` / `_filename_to_object_key()` to include `user_id`; add per-user tracking |
| `lmcache/v1/distributed/l2_adapters/mooncake_store_l2_adapter.py` | Same per-user tracking pattern as mock adapter |
| `lmcache/v1/multiprocess/blend_server_v2.py` | Pass `user_id` to `ipc_key_to_object_keys()` in all 4 call sites (same pattern as `server.py`) |
| `csrc/storage_backends/fs/connector.cpp` | Update `key_to_filename()` parser to handle `user_id@` prefix in the `@`-separated key format |
| `lmcache/v1/distributed/storage_controllers/eviction_controller.py` | Accept `QuotaManager`; add per-user usage trigger in `_check_and_evict()` |
| `lmcache/v1/distributed/storage_manager.py` | Create `QuotaManager`; pass to `L2EvictionController`; expose for HTTP server |
| `lmcache/v1/multiprocess/http_server.py` | Add `PUT/GET/DELETE /api/quota/{user_id}` and `GET /api/quota` endpoints |
| `docs/design/l2_adapters/l2_eviction.md` | Update to document UserLRU policy and per-user quota configuration |

### No vLLM Core Changes Required

The API caller passes `user_id` via the existing `extra_args.kv_transfer_params`
mechanism. vLLM forwards `extra_args` to `SamplingParams` unchanged — no vLLM
code modifications are needed.

### Test Files

| File | Description |
|------|-------------|
| `tests/v1/distributed/test_user_lru_eviction_policy.py` (new) | Unit tests for `UserLRUEvictionPolicy` |
| `tests/v1/distributed/test_lru_eviction_policy.py` | Verify existing LRU tests still pass with new `user_id` param |
| `tests/v1/distributed/test_l2_eviction.py` | Integration test for per-user eviction with mock adapter |
