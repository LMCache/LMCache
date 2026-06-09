# `lmcache describe` — Design & Implementation Plan

**Status:** Proposal  |  **Date:** 2026-03-19

## Context

The CLI framework (Phase 0) is complete — `BaseCommand`, `Metrics`, `MockCommand`,
and entry point are all working. The next step (Phase 1 per
[commands.md](../commands.md)) is to implement `lmcache describe kvcache`, which
provides a rich status dashboard of a running LMCache KV cache service.

`describe engine` is Phase 2 scope and stubbed but not implemented here.

---

## Command UX

```bash
$ lmcache describe kvcache --url http://localhost:8000

============ LMCache KV Cache Service ============
Health:                                  OK
ZMQ endpoint:                            tcp://localhost:5555
HTTP endpoint:                           http://localhost:8000
Engine type:                             blend
Chunk size:                              256
L1 capacity (GB):                        60.00
L1 used (GB):                            42.30 (70.5%)
Eviction policy:                         LRU
Cached objects:                          1024
Active sessions:                         3
Uptime:                                  2h 14m 32s
------ Model: meta-llama/Llama-3.1-70B-Instruct ---
World size:                              4
GPU IDs:                                 0, 1, 2, 3
Num layers:                              80
Num blocks:                              2048
Cache size per token (bytes):            327680
--- Kernel group 0 (meta-llama/Llama-3.1-70B-Instruct) ---
Kernel group index:                      0
Engine group index:                      0
Object group index:                      0
Num layers:                              80
Physical block size:                     128
Compress ratio:                          1
Dtype:                                   torch.float16
MLA:                                     False
Attention backend:         vLLM non-MLA flash attention
GPU KV shape:              NL x [2, NB, BS, NH, HS]
GPU KV tensor shape:       80 x [2, 2048, 128, 8, 128]
----------- L2: NixlStoreL2Adapter ------------
Type:                          NixlStoreL2Adapter
Health:                                  OK
Backend:                                 nixl_rdma
Stored objects:                          512
Pool used:                     480 / 512 (93.8%)
==================================================
```

JSON output collects model and L2 adapter sections into lists for easier
programmatic access:

```json
{
  "title": "LMCache KV Cache Service",
  "metrics": {
    "health": "OK",
    "url": "http://localhost:8000",
    "engine_type": "blend",
    ...
    "models": [
      {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "world_size": 4,
        "gpu_ids": "0, 1, 2, 3",
        "num_layers": 80,
        "num_blocks": 2048,
        "cache_size_per_token": 327680
      }
    ],
    "kernel_groups": [
      {
        "model": "meta-llama/Llama-3.1-70B-Instruct",
        "kernel_group_idx": 0,
        "engine_group_idx": 0,
        "object_group_idx": 0,
        "num_layers": 80,
        "physical_block_size": 128,
        "compress_ratio": 1,
        "dtype": "torch.float16",
        "is_mla": false,
        "attention_backend": "vLLM non-MLA flash attention",
        "gpu_kv_shape": "NL x [2, NB, BS, NH, HS]",
        "gpu_kv_concrete_shape": "80 x [2, 2048, 128, 8, 128]"
      }
    ],
    "l2_adapters": [
      {
        "type": "NixlStoreL2Adapter",
        "health": "OK",
        "backend": "nixl_rdma",
        "stored_object_count": 512,
        "pool_used": "480 / 512 (93.8%)"
      }
    ]
  }
}
```

Per-model sections are generated for each unique `(model_name, world_size)` pair
registered with the engine. The model section carries the context-wide fields —
`num_layers`, `num_blocks`, and `cache_size_per_token` — and is followed by one
**kernel group** section per kernel group, since a hybrid model's groups can
differ in geometry.

Each kernel group section includes:

- **Kernel / engine / object group index** — the group's identity:
  `kernel_group_idx` enumerates the manager's kernel groups, `engine_group_idx`
  is the paged-block address space (0 for non-hybrid), and `object_group_idx` is
  the owning object group.
- **Num layers** and **Physical block size** — the group's layer count and
  `shape_desc.bs`.
- **Compress ratio** — logical tokens per physical slot (1 for non-compressed).
- **Dtype** and **MLA** — the group's torch dtype and MLA flag.
- **Attention backend** — which attention implementation is active (e.g.,
  `vLLM non-MLA flash attention`, `vLLM MLA`, `SGLang MHA`), derived from the
  `GPUKVFormat` enum.
- **GPU KV shape** — the symbolic tensor layout using short names matching the
  `GPUKVFormat` enum (NB=num_blocks, NL=num_layers, BS=block_size, NH=num_heads,
  HS=head_size, PBS=page_buffer_size). E.g., `NL x [2, NB, BS, NH, HS]`.
- **GPU KV tensor shape** — the same layout with actual numeric values substituted
  from the group's `shape_desc` (e.g., `80 x [2, 2048, 128, 8, 128]`), so it is
  group-accurate.

L2 adapter sections are generated for each adapter in
`storage_manager.l2_adapters`. Fields shown depend on the adapter type:

- **Type** and **Health** — always shown.
- **Backend** — Nixl adapter backend (e.g., `nixl_rdma`).
- **Base path** — filesystem adapter storage directory.
- **Stored objects** — number of cached objects (Mock, Nixl).
- **Used** — size used vs capacity with percentage (Mock).
- **Pool used** — pool slots used vs total with percentage (Nixl).

---

## Design Decisions

### 1. Sub-target as positional argument

```
lmcache describe kvcache --url http://localhost:8000
lmcache describe engine  --url http://localhost:8000   # (Phase 2)
```

Uses a positional `target` argument with `choices=["kvcache"]` (extend to
`"engine"` in Phase 2). Matches the `describe {kvcache,engine}` pattern in
[commands.md](../commands.md).

### 2. `--url` points to the HTTP endpoint

The original design doc example shows `--url localhost:5555` (ZMQ port), but also
states that `describe kvcache` "gathers data from ... `/status` (HTTP)". The
HTTP `/status` endpoint already exposes **all** data needed (engine type, chunk
size, L1 memory, eviction policy, cached objects, health, sessions, etc.). Using
HTTP as the sole data source keeps the CLI simple — no ZMQ client needed.

`--url` accepts the HTTP base URL (e.g., `http://localhost:8000`). The command
normalizes it (adds `http://` if missing) and appends `/status`.

### 3. Output fields mapped from `/status`

| Display label | Machine key | Source in `/status` response |
|---|---|---|
| Health | `health` | `is_healthy` → `"OK"` / `"UNHEALTHY"` |
| ZMQ endpoint | `zmq_endpoint` | `zmq_endpoint` **(new — see Server-Side Changes)** |
| HTTP endpoint | `http_endpoint` | `http_endpoint` **(new — see Server-Side Changes)** |
| Engine type | `engine_type` | `engine_type` |
| Chunk size | `chunk_size` | `chunk_size` |
| L1 capacity (GB) | `l1_capacity_gb` | `storage_manager.l1_manager.memory_total_bytes` / 1024^3 |
| L1 used (GB) | `l1_used_gb` | `storage_manager.l1_manager.memory_used_bytes` / 1024^3, with `memory_usage_ratio` × 100 for % |
| Eviction policy | `eviction_policy` | `storage_manager.eviction_controller.eviction_policy` |
| Cached objects | `cached_objects` | `storage_manager.l1_manager.total_object_count` |
| Active sessions | `active_sessions` | `active_sessions` |
| Uptime | `uptime` | `uptime_seconds` **(new — see Server-Side Changes)**, formatted as `Xh Ym Zs` |

### 4. HTTP client: stdlib `urllib`

No new dependencies. Uses `urllib.request` following the same pattern as the
existing `lmcache/tools/mp_status_viewer/__main__.py`.

### 5. Error handling

| Condition | Behavior |
|---|---|
| Connection refused / timeout | Print error to stderr, exit 1 |
| HTTP 503 (engine not initialized) | Print "Server unhealthy: engine not initialized", exit 1 |
| Missing fields in response | Display as `N/A` (Metrics default for `None` values) |

---

## Server-Side Changes

Three fields in the design doc's `describe kvcache` output are **not currently
available** from `/status`. The following changes surface them.

### 1. Add `start_time` to `MPCacheEngine` → expose `uptime_seconds`

**File:** `lmcache/v1/multiprocess/server.py`

`MPCacheEngine.__init__()` (line 147) records `self._start_time = time.monotonic()`
at construction. `report_status()` (line 696) includes a new field:

```python
"uptime_seconds": time.monotonic() - self._start_time,
```

The CLI formats this as a human-readable string (e.g., `2h 14m 32s`).

### 2. Pass endpoint addresses into `MPCacheEngine` → expose in status

**File:** `lmcache/v1/multiprocess/server.py`

Currently `MPCacheEngine` does not know the ZMQ or HTTP addresses — those live in
`MPServerConfig` and `HTTPFrontendConfig`, which are only available in
`run_cache_server()` / `run_http_server()`.

**Option A — engine constructor params:** Add optional `zmq_endpoint: str | None`
and `http_endpoint: str | None` kwargs to `MPCacheEngine.__init__()`. Callers
(`run_cache_server` at line 787, and the blend variant) pass these when available.
`report_status()` includes them.

**Option B — set after construction:** Add setter methods or attrs that
`run_cache_server()` / `run_http_server()` set after creating the engine, before
returning it. This avoids changing the constructor signature.

**Recommendation:** Option A is simpler and more explicit.

```python
# In run_cache_server() (line 787):
engine = MPCacheEngine(
    storage_manager_config=storage_manager_config,
    chunk_size=mp_config.chunk_size,
    hash_algorithm=mp_config.hash_algorithm,
    zmq_endpoint=f"tcp://{mp_config.host}:{mp_config.port}",
)

# In run_http_server() lifespan (line 77):
# After engine is created, set http_endpoint:
engine.http_endpoint = f"http://{http_config.http_host}:{http_config.http_port}"
```

Note: The ZMQ endpoint is known at `run_cache_server()` time, but the HTTP
endpoint is only known in `run_http_server()`. Since `run_http_server()` calls
`run_cache_server(return_engine=True)` and gets back the engine, it can set
`http_endpoint` after construction. So a hybrid approach works:
- `zmq_endpoint` passed via constructor (always available)
- `http_endpoint` set as an attribute after construction (only when HTTP frontend
  is enabled)

`report_status()` returns both:

```python
"zmq_endpoint": self.zmq_endpoint,
"http_endpoint": getattr(self, "http_endpoint", None),
```

### 3. Same changes for `BlendCacheEngine`

**File:** `lmcache/v1/multiprocess/blend_server.py` (and `blend_server_v2.py`)

Mirror the same `start_time`, `zmq_endpoint`, and `http_endpoint` additions if
`BlendCacheEngine` has its own `report_status()`. If it delegates to
`MPCacheEngine`, no separate change is needed.

### Summary of server-side changes

| Field | Where | Change |
|---|---|---|
| `uptime_seconds` | `MPCacheEngine.__init__` + `report_status()` | Record `time.monotonic()` at init, compute delta in status |
| `zmq_endpoint` | `MPCacheEngine.__init__` + `run_cache_server()` | New constructor kwarg, passed from `MPServerConfig` |
| `http_endpoint` | `run_http_server()` lifespan + `report_status()` | Set on engine after construction when HTTP is enabled |

### 4. Expose GPU KV format, shape, and attention backend in `kv_cache_layout`

**Files:** `lmcache/v1/gpu_connector/utils.py`, `lmcache/v1/multiprocess/gpu_context.py`, `lmcache/v1/multiprocess/server.py`

Helper functions in `utils.py` (derived from `legible_print_gpu_kv_format()`):
- `get_gpu_kv_shape_description(gpu_kv_format)` — symbolic shape (e.g., `NL x [2, NB, BS, NH, HS]`)
- `get_attention_backend(gpu_kv_format)` — backend name (e.g., `vLLM non-MLA flash attention`)
- `get_concrete_gpu_kv_shape(kv_caches, gpu_kv_format)` — whole-context shape with actual values
- `get_concrete_gpu_kv_shape_from_shape_desc(shape_desc, gpu_kv_format)` — **group-accurate** shape with actual values, read from a single kernel group's `PageBufferShapeDesc` (used by `report_status`)

`report_status()` is organised **per kernel group**: a small set of context-wide
fields at the top level, plus a `kernel_groups` list where each entry is
self-describing. The format-derived fields (`gpu_kv_format`, `gpu_kv_shape`,
`attention_backend`, `is_mla`) and the group-accurate `gpu_kv_concrete_shape`
live inside each group:

```python
"kv_cache_layout": {
    "num_layers": 80,
    "inference_engine_logical_block_size": 128,
    "num_blocks": 2048,
    "cache_size_per_token": 327680,
    "kernel_groups": [
        {
            "kernel_group_idx": 0,
            "engine_group_idx": 0,
            "object_group_idx": 0,
            "num_layers": 80,
            "layer_indices": [0, 1, ...],
            "physical_block_size": 128,
            "compress_ratio": 1,
            "dtype": "torch.float16",
            "gpu_kv_concrete_shape": "80 x [2, 2048, 128, 8, 128]",
            "is_mla": false,
            "gpu_kv_format": "NL_X_TWO_NB_BS_NH_HS",
            "gpu_kv_shape": "NL x [2, NB, BS, NH, HS]",
            "attention_backend": "vLLM non-MLA flash attention",
        },
    ],
}
```

---

## CLI Implementation

### New file: `lmcache/cli/commands/describe.py`

```python
class DescribeCommand(BaseCommand):
    name() → "describe"
    help() → "Show detailed status of a running LMCache service."

    add_arguments(parser):
        parser.add_argument("target", choices=["kvcache"],
                            help="What to describe.")
        parser.add_argument("--url", default="http://localhost:8080",
                            help="LMCache HTTP server URL")

    execute(args):
        if args.target == "kvcache":
            self._describe_kvcache(args)

    _describe_kvcache(args):
        1. Normalize URL (ensure http:// prefix)
        2. Fetch JSON from {url}/status (timeout=10s)
        3. On error: print to stderr, sys.exit(1)
        4. Extract fields from nested response dict
        5. Format uptime_seconds → "Xh Ym Zs"
        6. Format L1 used bytes → "XX.XX (YY.Y%)"
        7. Build flat Metrics via self.create_metrics() (width=48)
        8. metrics.emit()
```

Module-level helpers:

```python
def _fetch_json(url: str, timeout: int = 10) -> dict:
    """GET *url*, return parsed JSON. Raises on HTTP/network errors."""

def _normalize_url(url: str) -> str:
    """Ensure URL has http:// scheme, strip trailing slash."""

def _fmt_uptime(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs'."""

def _fmt_used_gb(used_bytes: int, ratio: float) -> str:
    """Format as 'XX.XX (YY.Y%)'."""
```

### Modify: `lmcache/cli/commands/__init__.py`

Add import and registry entry:

```python
from lmcache.cli.commands.describe import DescribeCommand

ALL_COMMANDS: list[BaseCommand] = [
    MockCommand(),
    DescribeCommand(),
]
```

### Patterns to follow

- **Reuse `BaseCommand.create_metrics()`** — auto-handles `--format` and `--output`
  flags (see `base.py`).
- **Flat metrics, no sections** — all fields at top level, matching the design doc
  output style. Use `metrics.add(key, label, value)` directly.
- **Width = 48** — matches the divider width in `commands.md` examples.

---

## Verification

1. **Unit test:** Test `_normalize_url()`, `_fmt_uptime()`, `_fmt_used_gb()`, and
   field extraction logic with a synthetic `/status` response dict (no live
   server needed).
2. **Manual test against running server:**
   ```bash
   lmcache describe kvcache --url http://localhost:8000
   lmcache describe kvcache --url http://localhost:8000 --format json
   lmcache describe kvcache --url localhost:8000          # auto-prefix http://
   lmcache describe kvcache --url http://localhost:8000 --output status.json
   lmcache describe kvcache --url http://localhost:9999   # connection refused → exit 1
   ```
3. **JSON output:** Verify machine keys are snake_case and values are raw types
   (not display-formatted strings), except `l1_used_gb` and `uptime` which include
   human-readable formatting.

---

## Future Work (out of scope)

- `describe engine` (Phase 2) — queries vLLM's `/v1/models` and `/health`
  endpoints for model name, context length, running requests.
