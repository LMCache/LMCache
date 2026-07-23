# SGLang PD (Prefill/Decode) Disaggregation over NIXL

## Summary

This is a first-class LMCache feature for **prefill/decode (PD) disaggregation**
in SGLang: a pool of prefill instances computes KV caches and pushes them,
per request, to a pool of decode instances over NIXL, so decode never recomputes
the prompt. It is *not* built on SGLang's HiCache decode-restore path — LMCache
owns the transfer end to end via its existing `PDBackend`.

The transport already exists. LMCache's `lmcache.v1.storage_backend.pd_backend.PDBackend`
implements a complete NIXL-based cross-instance KV push (sender/receiver roles,
side-channel handshake, remote allocation, proxy notification). What was missing
was the SGLang glue: a way for a prefill worker to say *"store this request's KV
**and** ship it to decode instance X"*. That is a single new piece of
per-request routing state — `transfer_spec` — plus a connector that forwards it.

The design supports **xPyD** (N prefill × M decode) from the start: routing is
per request, so any prefill can target any decode; there is no 1:1 pinning.

It runs on **Intel XPU** as well as CUDA — the LMCache side uses
`torch_device_type` throughout, and the PD buffer device is configurable
(`pd_buffer_device: xpu`).

## Goals / Non-Goals

Goals:

- A dedicated `LMCachePDConnector` that pushes a completed request's KV to its
  assigned decode peer, driven entirely by LMCache's `PDBackend`.
- Per-request routing (`DisaggSpec`) injected by the router/proxy through
  SGLang's request path — enabling xPyD with no static prefill↔decode pinning.
- Full backward compatibility: a request with no routing (`transfer_spec=None`)
  stores exactly as before (local CPU/disk/remote offload only).
- Works on XPU and CUDA; the PD buffer device is configurable.

Non-Goals:

- Reusing SGLang's HiCache decode-side restore machinery (explicitly out).
- Building a new transport — `PDBackend`/NIXL is reused unchanged.
- The router/proxy implementation itself (xPyD scheduling policy) — this doc
  covers only the data-plane contract the proxy must satisfy.
- Layerwise streaming of the PD push (see *Why non-layerwise* below).

## Types

- **`DisaggSpec`** (`lmcache/integration/sglang/pd_types.py`, `@dataclass`):
  per-request routing to the decode receiver. Deliberately dependency-free (no
  `sglang`/`torch` imports) so it can be constructed and unit-tested without a
  GPU or a full SGLang install; `sglang_adapter` re-exports it.
  - `req_id: str` — correlation id shared with the proxy; returned in the
    `ProxyNotif` when the last-prefill transfer completes.
  - `receiver_host: str` — decode instance host/IP.
  - `receiver_init_port: List[int]` — NIXL side-channel init ports, **indexed by
    the sender's tp rank** (one listen port per receiver rank).
  - `receiver_alloc_port: List[int]` — remote-allocation ports, indexed by tp rank.
  - `receiver_query_port: List[int]` — cache-query ports, indexed by tp rank;
    used only when bidirectional query is enabled, empty otherwise.
  - `is_last_prefill: bool` — `True` on the final store for a request (triggers
    the proxy notification). Single-shot stores set this `True`.
  - `from_dict(spec) -> DisaggSpec` — parses/validates the proxy-injected
    `kv_transfer_params["disagg_spec"]` mapping; raises `ValueError` on a missing
    required key or a non-`list[int]` port field.

  The port fields are lists indexed by tp rank because `PDBackend`, running on
  the sender at rank `tp_rank`, reads `receiver_init_port[self.tp_rank]` (and the
  alloc/query equivalents) to reach the *same* rank on the receiver. This
  matches the field contract already consumed by
  `PDBackend.batched_submit_put_task`.

- **`StoreMetadata`** (`sglang_adapter.py`): gains
  `transfer_spec: Optional[DisaggSpec] = None`. Default `None` keeps every
  existing (non-PD) caller unchanged.

## Connector

`LMCachePDConnector(LMCacheConnector)` (`sglang_adapter.py`):

- On `store_kv`, forwards `store_metadata.transfer_spec` into
  `LMCacheEngine.store(..., transfer_spec=...)`. The storage manager routes it to
  `PDBackend`, which performs the NIXL push to the decode peer. `transfer_spec is
  None` degrades to a plain local store.
- Logs at `info` when a push is issued (req_id, token count, receiver host) and
  at `debug` when there is no routing.

### Why non-layerwise

`LMCachePDConnector` extends the **non-layerwise** `LMCacheConnector`, not
`LMCacheLayerwiseConnector`, because **only** the non-layerwise
`LMCacheEngine.store` path forwards `transfer_spec` to the storage manager.
`store_layer` (the layerwise path) does not thread `transfer_spec` through, so a
layerwise PD sender would silently drop the routing and never push. This is the
key architectural constraint of the feature: the PD push is a whole-request
store, issued once the request's KV is complete.

## Data flow

```
proxy/router                 prefill worker (sender)              decode worker (receiver)
     |                              |                                     |
  pick decode peer,           req arrives with                           |
  build disagg_spec  ───────► kv_transfer_params                         |
  (req_id, host, ports)       .disagg_spec                               |
     |                              |                                     |
     |                        request finishes:                          |
     |                        cache_finished_req builds                  |
     |                        StoreMetadata(transfer_spec=               |
     |                          DisaggSpec.from_dict(...))                |
     |                              |                                     |
     |                        LMCachePDConnector.store_kv                 |
     |                              │ engine.store(transfer_spec=…)       |
     |                              ▼                                     |
     |                          PDBackend ── NIXL push (per tp rank) ───► KV buffer
     |                              |          init/alloc handshake       │ populated
     |◄─── ProxyNotif(req_id) ──────┘  (on is_last_prefill)               │
     |                                                                    ▼
  route req to decode ──────────────────────────────────────────► decode reads local KV
```

The port lists let each sender rank talk to the matching receiver rank, so the
push parallelizes across tensor-parallel ranks.

## SGLang-side plumbing (fork changes, out of this repo)

The proxy injects routing as `kv_transfer_params={"disagg_spec": {...}}` on the
request. To reach `cache_finished_req`, the fork must thread `kv_transfer_params`
through: `GenerateReqInput` / `TokenizedGenerateReqInput` (`io_struct.py`),
`tokenizer_manager.py`, `Req.__init__` (`schedule_batch.py`), the `Req` build in
`scheduler.py`, and finally `LMCRadixCache.cache_finished_req`
(`lmc_radix_cache.py`), which builds the `StoreMetadata` and calls the connector.
`server_args.py` gains the flags to select `LMCachePDConnector` and point at the
LMCache PD config. These are tracked separately from this LMCache PR.

## Configuration

Sender (prefill) and receiver (decode) each run an LMCache engine with PD
enabled (`lmcache/v1/config.py`):

- `enable_pd: true`
- `pd_role: sender` (prefill) / `receiver` (decode)
- `pd_buffer_size`, `pd_buffer_device: xpu` (or `cuda`)
- `pd_peer_host`, `pd_peer_init_port`, `pd_peer_alloc_port`, `pd_peer_query_port`
- `pd_backend_mode: sync | async`, `pd_bidirectional`

The receiver sets `remove_after_retrieve` automatically. Legacy aliases
(`enable_xpyd`→`enable_pd`, `nixl_role`→`pd_role`, `nixl_peer_*`→`pd_peer_*`)
remain accepted.

## Testing

`tests/v1/test_sglang_pd_disagg.py`:

- `DisaggSpec.from_dict` parsing/validation runs **without** sglang/torch by
  importing from `pd_types` directly (required fields, optional fields, missing
  key raises, non-`list[int]` ports raise, port lists copied not aliased).
- Connector forwarding (`transfer_spec` reaches `engine.store`; `None` still
  stores locally) is guarded by `importorskip("...sglang_adapter")` since it
  needs the sglang import; it skips cleanly in the LMCache CI image and runs in
  the SGLang container.

## Backward compatibility

`transfer_spec` defaults to `None` everywhere. Non-PD SGLang deployments, and
every existing connector (`LMCacheConnector`, `LMCacheLayerwiseConnector`,
`LMCacheMPConnector`), are unaffected: with no `DisaggSpec`, `store` takes the
identical local-offload path it always has.
