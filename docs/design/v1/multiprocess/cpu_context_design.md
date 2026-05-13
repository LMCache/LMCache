# CPU Context Design (MP mode, non-CUDA)

## Scope

This document describes the non-CUDA CPU-based KV transfer path for LMCache
multiprocess mode.

The goal is to support KV transfer for non-CUDA devices (for example CPU,
XPU, HPU) without changing the existing CUDA IPC path, while providing a
clean abstraction layer that makes it easy to add alternative transport
mechanisms (e.g. shared memory) in a future PR.

## Why this path exists

The CUDA path uses IPC wrappers around GPU tensors and the existing
`REGISTER_KV_CACHE` / `STORE` / `RETRIEVE` request flow.

For non-CUDA tensors, CUDA IPC is not available.  The CPU context path
provides a generic protocol where workers:

1. Gather KV blocks into CPU chunk(or memory obj) tensors.
2. Transport those CPU chunks to the server storage through a concrete
   `CPUContext` implementation.
3. Retrieve CPU chunks(or memory obj) from the server and scatter them back into device KV
   tensors.

## Protocol additions

Three request types are used for non-CUDA mode (unchanged from the original
cpu context design):

- `REGISTER_KV_CACHE_CPU_CONTEXT`
- `STORE_CPU_CHUNKS`
- `RETRIEVE_CPU_CHUNKS`

These are registered in the MP server dispatch and have corresponding
payload/response contracts in the multiprocess protocol definitions.

## File structure

```
lmcache/v1/multiprocess/
├── cpu_context.py         # CPUContextMetadata, CPUContext(ABC), factory, gather/scatter utils
└── cpu_context_pickle.py  # CPUContextPickle — pickle-based concrete implementation
```

### `cpu_context.py`

Provides:

- **`CPUContextMetadata`** dataclass — layout metadata:

  ```python
  @dataclass
  class CPUContextMetadata:
      layout_desc: MemoryLayoutDesc
      block_size: int
      use_mla: bool
  ```

- **`CPUContext(ABC)`** — abstract base class with `mq_client` as a common
  dependency.  All concrete implementations share the same two-phase
  `prepare/commit` interface:

  ```python
  class CPUContext(ABC):
      def __init__(self, metadata: CPUContextMetadata, mq_client, mq_timeout: float): ...

      @abstractmethod
      def prepare_store(self, key, instance_id, chunks: list[torch.Tensor]) -> Any: ...
      @abstractmethod
      def commit_store(self, handle: Any) -> bool: ...
      @abstractmethod
      def prepare_retrieve(self, key, instance_id) -> tuple[Any, list[torch.Tensor] | None]: ...
      @abstractmethod
      def commit_retrieve(self, handle: Any) -> None: ...
      @abstractmethod
      def close(self) -> None: ...
  ```

- **`create_cpu_context()`** factory — currently always returns a
  `CPUContextPickle` instance; a future SHM-capable PR can extend this to
  probe for shared-memory availability and fall back to pickle.

- **Shared utility functions** used by all concrete implementations:
  - `compute_kv_layout` — extract block size, layer count, hidden dim and
    dtype from live KV tensors.
  - `gather_paged_kv_to_cpu` — gather paged KV blocks into a list of CPU
    tensors (one per LMCache chunk).
  - `scatter_cpu_to_paged_kv` — scatter CPU chunk tensors back into paged
    KV tensors.

### `cpu_context_pickle.py`

Provides **`CPUContextPickle(CPUContext)`**:

| Phase | What happens |
|---|---|
| `prepare_store` | `pickle.dumps(chunks)` → returns `(key, instance_id, bytes)` as opaque handle |
| `commit_store` | sends `STORE_CPU_CHUNKS` via `mq_client`, blocks for server ack, returns `bool` |
| `prepare_retrieve` | sends `RETRIEVE_CPU_CHUNKS` via `mq_client`, blocks for response, `pickle.loads` → returns `(None, chunks)` or `(None, None)` on miss |
| `commit_retrieve` | no-op (pickle path holds no server-side locks) |
| `close` | no-op |

## Tensor/chunk contracts

Chunk formats are unchanged:

- non-MLA: `[2, num_layers, chunk_tokens, hidden_dim]`
- MLA: `[num_layers, chunk_tokens, hidden_dim]`

Internal gather/scatter uses block-level indexing to avoid token-level slot
expansion and token-wise select/copy operations.

## Layout handling

Supported KV formats in CPU gather/scatter:

- `NL_X_TWO_NB_BS_NH_HS` (NHD)
- `NL_X_NB_TWO_BS_NH_HS` (NHD flashinfer)
- `NL_X_TWO_NB_NH_BS_HS` (HND)
- `NL_X_NB_TWO_NH_BS_HS` (HND flashinfer)
- `NL_X_NB_BS_HS` (MLA)

## Worker adapter integration

The adapter now delegates transport behavior to
`lmcache/v1/multiprocess/transfer_context.py`.

`create_transfer_context(kv_caches)` centralizes device dispatch:

- all CUDA → existing CUDA IPC registration and store/retrieve path
- all non-CUDA → `CPUTransferContext` with cpu context registration and CPU context store/retrieve path

`LMCacheMPSchedulerAdapter` now holds `self.transfer_ctx: TransferContext | None`
and calls:

- `transfer_ctx.register(...)`
- `transfer_ctx.submit_store(...)`
- `transfer_ctx.submit_retrieve(...)`
- `transfer_ctx.poll_finished()` (healthy) or `transfer_ctx.drain_all()` (unhealthy)

### Store path (non-CUDA)

```python
# CPUTransferContext.submit_store
cpu_chunks = gather_paged_kv_to_cpu(kv_caches, block_ids, blocks_in_chunk, ...)
handle = self._cpu_context.prepare_store(key, instance_id, cpu_chunks)
ok = self._cpu_context.commit_store(handle)   # synchronous; blocks for server ack
self._store_done[request_id] = ok
```

`CPUTransferContext.poll_finished()` drains `_store_done` on each call.

### Retrieve path (non-CUDA)

```python
# CPUTransferContext.submit_retrieve
handle, chunks = self._cpu_context.prepare_retrieve(key, instance_id)  # synchronous
ok = chunks is not None
if chunks is not None:
    try:
        scatter_cpu_to_paged_kv(kv_caches, block_ids, chunks, blocks_in_chunk,
                                skip_first_n_tokens=skip_first_n_tokens, ...)
    except (RuntimeError, ValueError, TypeError, IndexError):
        ok = False
self._cpu_context.commit_retrieve(handle)
self._retrieve_done[request_id] = (ok, block_ids)
```

`CPUTransferContext.poll_finished()` drains `_retrieve_done` on each call.
The adapter passes `op.skip_first_n_tokens` into
`transfer_ctx.submit_retrieve(..., skip_first_n_tokens=...)`.

The retrieve is **synchronous inside `CPUTransferContext.submit_retrieve`**;
`poll_finished()` just drains request ids recorded by submit methods.

## Server integration

`MPCacheEngine` holds:

- `cpu_contexts: dict[int, CPUContextMetadata]` — per-instance metadata.
- `cpu_context_meta: dict[int, tuple[str, int]]` — per-instance
  `(model_name, world_size)` for layout resolution.

Server-side handler methods are unchanged:
- `register_kv_cache_cpu_context` — stores `CPUContextMetadata` in `cpu_contexts`.
- `store_cpu_chunks` — unpickles payload, copies tensors into storage.
- `retrieve_cpu_chunks` — reads from storage, pickles tensors, returns bytes.

Additional integration points:

- Unregister cleanup removes both `cpu_contexts` and `cpu_context_meta`.
- Layout lookup via `_find_layout_desc` resolves both GPU and CPU context
  registrations.
- Status reporting (`report_status`) includes `registered_cpu_instance_ids`
  and `cpu_context_meta`.

## CUDA vs non-CUDA state machine

```text
                           register_kv_caches()
                                      |
                                      v
                    create_transfer_context(kv_caches)
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
              [device == cuda]                 [device != cuda]
                     |                                 |
                     v                                 v
      CudaTransferContext.register()      CPUTransferContext.register()
      REGISTER_KV_CACHE (CUDA IPC)        REGISTER_KV_CACHE_CPU_CONTEXT (CPU metadata)
                     |                         + create_cpu_context()
                     +----------------+----------------+
                                      |
                                      v
                              [READY / SERVING]
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
        transfer_ctx.submit_store()       transfer_ctx.submit_store()
                     |                                 |
                     v                                 v
             STORE (GPU -> L1)           gather_paged_kv_to_cpu()
                      |                 + _cpu_context.prepare_store()
                      v                 + _cpu_context.commit_store()  [sync]
                  [READY]                       _store_done[id] = ok
                     |                                 |
                     +----------------+----------------+
                                      |
                                      v
      transfer_ctx.submit_retrieve() + get_finished()
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
           RETRIEVE (L1 -> GPU)    _cpu_context.prepare_retrieve()  [sync]
           [async future]          + scatter_cpu_to_paged_kv()
                                   + _cpu_context.commit_retrieve()
                                   _retrieve_done[id] = (ok, block_ids)
                     |                                 |
                     +----------------+----------------+
                                      |
                                      v
                                [READY / SERVING]
                                      |
                                      v
                           unregister_kv_cache()
                                      |
                                      v
                                  [TERMINATED]
```

## Future extension: CPUContextShm

The `CPUContext` base class is designed to accommodate a shared-memory
implementation in a future PR with minimal changes:

| Phase | Pickle | SHM (future) |
|---|---|---|
| `prepare_store` | `pickle.dumps` | MQ `PREPARE_STORE` → slot metadata → memcpy |
| `commit_store` | MQ `STORE_CPU_CHUNKS` | MQ `COMMIT_STORE` |
| `prepare_retrieve` | MQ `RETRIEVE_CPU_CHUNKS` + `pickle.loads` | MQ `PREPARE_RETRIEVE` → tensor views from SHM |
| `commit_retrieve` | no-op | MQ `FINISH_READ` (release read lock) |

The `create_cpu_context()` factory will probe for SHM availability and fall
back to pickle when SHM is unavailable.

## Validation coverage

`tests/v1/multiprocess/test_cpu_context.py` covers:

- CPU wrapper behavior (`wrap_kv_caches` with cpu context mode)
- NHD and MLA gather/scatter round-trip
- HND round-trip for both HND formats
- `skip_first_n_tokens` behavior
- Server-side register/store/retrieve flow

`tests/v1/test_vllm_mp_adapter.py` covers transfer-context integration,
including CPU registration path (`REGISTER_KV_CACHE_CPU_CONTEXT`) and
store/retrieve submit delegation.

## Non-goals

- No change to existing CUDA IPC path semantics.
- No CPU-specific logic added to shared `gpu_connector/utils.py`.
