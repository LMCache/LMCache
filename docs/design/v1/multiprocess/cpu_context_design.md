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

1. Gather KV blocks into CPU chunk tensors.
2. Transport those CPU chunks to the server storage through a concrete
   `CPUContext` implementation.
3. Retrieve CPU chunks from the server and scatter them back into device KV
   tensors.

## Protocol additions

Three request types are used for non-CUDA mode (unchanged from the original
cpu context design):

- `REGISTER_KV_CACHE_BOUNCE`
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

- **`CPUContextMetadata`** dataclass — layout metadata (replaces the old
  `CPUBounceContext` dataclass):

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
  - `gather_chunks_to_cpu` — gather paged KV blocks into a list of CPU
    tensors (one per LMCache chunk).
  - `scatter_cpu_chunks_to_kv` — scatter CPU chunk tensors back into paged
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

`lmcache/integration/vllm/vllm_multi_process_adapter.py` chooses the path
by tensor `device.type`:

- all CUDA → existing CUDA IPC registration and store/retrieve path
- all non-CUDA → cpu context registration and CPU context store/retrieve path

The adapter holds a `cpu_context: CPUContext` instance and uses the uniform
`prepare/commit` interface for both store and retrieve.

### Store path (non-CUDA)

```python
# submit_store_request
cpu_chunks = gather_chunks_to_cpu(kv_caches, block_ids, blocks_in_chunk, ...)
handle = self.cpu_context.prepare_store(key, instance_id, cpu_chunks)
ok = self.cpu_context.commit_store(handle)   # synchronous; blocks for server ack
self._cpu_store_done[request_id] = ok
```

`get_finished` drains `_cpu_store_done` on each call.

### Retrieve path (non-CUDA)

```python
# submit_retrieve_request
handle, chunks = self.cpu_context.prepare_retrieve(key, instance_id)  # synchronous
if chunks is not None:
    scatter_cpu_chunks_to_kv(kv_caches, block_ids, chunks, blocks_in_chunk,
                             skip_first_n_tokens=op.skip_first_n_tokens, ...)
self.cpu_context.commit_retrieve(handle)
self._cpu_retrieve_done[request_id] = (chunks is not None, block_ids)
```

`get_finished` drains `_cpu_retrieve_done` on each call.

The retrieve is now **synchronous in `submit_retrieve_request`**; there is no
separate future to poll.  This simplifies `get_finished` which no longer
needs a `if self._use_cpu_context:` branch for retrieve futures.

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
                             [Inspect device.type]
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
              [device == cuda]                 [device != cuda]
                     |                                 |
                     v                                 v
       REGISTER_KV_CACHE (CUDA IPC)      REGISTER_KV_CACHE_BOUNCE (CPU metadata)
                     |                         + create_cpu_context()
                     +----------------+----------------+
                                      |
                                      v
                              [READY / SERVING]
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
                submit_store()                    submit_store()
                     |                                 |
                     v                                 v
            STORE (GPU -> L1)           gather_chunks_to_cpu()
                     |                 + cpu_context.prepare_store()
                     v                 + cpu_context.commit_store()  [sync]
                 [READY]                    _cpu_store_done[id] = ok
                     |                                 |
                     +----------------+----------------+
                                      |
                                      v
                submit_retrieve() + get_finished()
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
          RETRIEVE (L1 -> GPU)    cpu_context.prepare_retrieve()  [sync]
          [async future]          + scatter_cpu_chunks_to_kv()
                                  + cpu_context.commit_retrieve()
                                  _cpu_retrieve_done[id] = (ok, block_ids)
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

## Non-goals

- No change to existing CUDA IPC path semantics.
- No CPU-specific logic added to shared `gpu_connector/utils.py`.
