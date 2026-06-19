# Implementierungsauftrag: Engine-Driven Multi-Group KV Cache Transfer für LMCache

## Ziel

Implementiere Unterstützung für den **engine-driven Transfer-Modus** bei hybriden Modellen mit
mehreren KV-Cache-Gruppen (GDN/Mamba + Attention, z. B. Qwen3.6-27B). Dadurch wird kein VRAM mehr
im `lmcache server`-Prozess benötigt: Statt dass der Server direkt auf GPU-Speicher zugreift,
führt der vLLM-Worker die GPU→CPU-Kopie selbst durch und schickt die fertigen CPU-Daten.

**Repository:** `https://github.com/LMCache/LMCache` · Branch `dev`  
**Sprache:** Python 3.12, keine externen Abhängigkeiten außer den bereits vorhandenen (`torch`,
`msgspec`, `pickle`, `numpy`).

---

## Hintergrund und Motivation

### Aktuelles Problem

Der `lmcache server`-Prozess belegt bei Qwen3.6-27B **666 MB VRAM pro GPU**, obwohl er eigentlich
ein reiner CPU-Caching-Daemon ist. Ursache ist der **lmcache-driven Transfer-Pfad**:

```
vLLM Worker (GPU)
  └─ sendet CUDA-IPC-Handle-Liste via REGISTER_KV_CACHE
  └─ sendet Block-IDs + CUDA-Event via STORE / RETRIEVE

LMCache Server (läuft in separatem Prozess)
  └─ öffnet CUDA-IPC-Handles → erzeugt CUDA Primary Context (~550 MB pro GPU)
  └─ alloziert block_ids_buffer (8 MB, 1M × int64)
  └─ alloziert _TempGPUBuffer Staging-Buffer (~5 MB)
  └─ führt multi_layer_block_kv_transfer CUDA-Kernel aus (GPU→CPU-Kopie)
  └─ speichert Ergebnis in L1-RAM
```

Der **engine-driven Transfer-Pfad** würde das GPU-Problem lösen:

```
vLLM Worker (GPU)
  └─ führt GPU→CPU-Kopie SELBST aus (gather_paged_kv_to_cpu)
  └─ sendet CPU-Bytes via COMMIT_STORE

LMCache Server (kein CUDA!)
  └─ empfängt CPU-Bytes, deserialisiert
  └─ speichert in L1-RAM
```

### Warum es für Hybrid-Modelle noch nicht funktioniert

Die Funktion `_single_group_block_ids` in
`lmcache/v1/multiprocess/transfer_context/worker_transfer.py` sperrt den engine-driven Pfad explizit
für alle Modelle mit mehr als einer KV-Cache-Gruppe:

```python
def _single_group_block_ids(block_ids: list[list[int]]) -> list[int]:
    """Return the flat block-id list for transports without HMA support."""
    if len(block_ids) != 1:
        raise RuntimeError(
            "engine-driven transfer does not support hybrid KV cache groups"
        )
    return block_ids[0]
```

Qwen3.6-27B hat mindestens zwei Gruppen (Attention + GDN-State), daher `len(block_ids) == 2` →
`RuntimeError` beim ersten Store/Retrieve.

---

## Zu ändernde Dateien

```
lmcache/v1/multiprocess/custom_types.py
lmcache/v1/multiprocess/transfer_context/base.py
lmcache/v1/multiprocess/transfer_context/worker_transfer.py
lmcache/v1/multiprocess/modules/engine_driven_transfer.py
lmcache/v1/multiprocess/modules/server_transfer.py          (nur minimale Änderung)
```

---

## Detaillierte Datenfluss-Analyse (Ist-Zustand, Einzel-Gruppe)

Um zu verstehen, was geändert werden muss, hier der vollständige Fluss für eine Einzel-Gruppe:

### A) Registrierung (Worker → Server)

**Worker** (`EngineDrivenTransferContext.register` in `worker_transfer.py`):
```python
block_size, num_layers, hidden_dim_size, dtype_str, engine_kv_format = compute_kv_layout(
    kv_caches, layout_hints=layout_hints
)
# Sendet RegisterEngineDrivenContextPayload via MQ an Server
future = send_request(mq_client, RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT, [
    RegisterEngineDrivenContextPayload(
        instance_id=instance_id, model_name=model_name, world_size=world_size,
        block_size=block_size, num_layers=num_layers,
        hidden_dim_size=hidden_dim_size, dtype_str=dtype_str, use_mla=use_mla_flag,
    )
])
```

**Server** (`EngineDrivenTransferModule.register_kv_cache_engine_driven_context` in
`engine_driven_transfer.py`):
```python
def register_kv_cache_engine_driven_context(self, payload: RegisterEngineDrivenContextPayload):
    shape = torch.Size([2, payload.num_layers, chunk_size, payload.hidden_dim_size])
    layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
    metadata = EngineDrivenContextMetadata(layout_desc=layout_desc, ...)
    self._engine_driven_contexts[payload.instance_id] = EngineDrivenContextEntry(metadata=..., ...)
    self._ctx.layout_desc_registry.register(payload.model_name, payload.world_size, layout_desc)
```

### B) Store (Worker → Server, Einzel-Gruppe)

**Worker** (`EngineDrivenTransferContext.submit_store` in `worker_transfer.py`):
```python
def submit_store(self, ..., kv_caches, block_ids, ...):
    torch_dev.synchronize()
    cpu_chunks = gather_paged_kv_to_cpu(
        kv_caches,
        _single_group_block_ids(block_ids),   # ← hier bricht Hybrid ab
        blocks_in_chunk,
        layout_hints=self._layout_hints,
        engine_kv_format=self._engine_kv_format,
    )
    ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)
```

**Serialisierung** (`PickleContext.commit_store` in
`transfer_context/pickle.py`):
```python
cpu_data = pickle.dumps(cpu_chunks)   # list[torch.Tensor]
send_request(mq_client, RequestType.COMMIT_STORE, [key, instance_id, cpu_data])
```

**Server** (`EngineDrivenTransferModule.commit_store` in `engine_driven_transfer.py`):
```python
def commit_store(self, key: IPCCacheServerKey, instance_id: int, cpu_data: bytes) -> bool:
    entry = self._engine_driven_contexts[instance_id]
    strategy = self._strategies[instance_id]
    strategy.commit_store(
        key=key, instance_id=instance_id, cpu_data=cpu_data,
        context=entry.metadata,
        resolve_obj_keys=lambda k: self._resolve_single_group_obj_keys(k)
    )
```

**Storage** (`PickleTransferStrategy.commit_store` in `server_transfer.py`):
```python
def commit_store(self, key, instance_id, cpu_data, context, resolve_obj_keys):
    obj_keys = resolve_obj_keys(key)           # für Object Group 0
    chunks: list[torch.Tensor] = pickle.loads(cpu_data)
    reserved = storage_manager.reserve_write(obj_keys, context.layout_desc, "new")
    for obj_key, chunk in zip(obj_keys, chunks):
        memory_obj = reserved[obj_key]
        memory_obj.raw_data[:chunk.nbytes] = chunk.numpy().tobytes()
    storage_manager.finish_write(list(reserved.keys()))
```

### C) Relevante Typen (Referenz)

```python
# custom_types.py
class RegisterEngineDrivenContextPayload(msgspec.Struct):
    instance_id: int
    model_name: str
    world_size: int
    block_size: int          # ← Einzel-Wert, muss zu Liste werden
    num_layers: int          # ← Einzel-Wert, muss zu Liste werden
    hidden_dim_size: int     # ← Einzel-Wert, muss zu Liste werden
    dtype_str: str           # ← Einzel-Wert, muss zu Liste werden
    use_mla: bool            # ← Einzel-Wert, muss zu Liste werden

# transfer_context/base.py
@dataclass
class EngineDrivenContextMetadata:
    layout_desc: MemoryLayoutDesc   # ← Einzel-Gruppe, muss zu Liste werden
    block_size: int                  # ← Einzel-Wert, muss zu Liste werden
    use_mla: bool                    # ← Einzel-Wert, muss zu Liste werden

# group_view.py
class EngineGroupInfo(msgspec.Struct, frozen=True):
    engine_group_id: int
    layer_indices: tuple[int, ...]   # Welche Schichten gehören dieser Gruppe
    tokens_per_block: int            # Blockgröße in Tokens (0 = unbekannt)
    sw_size_tokens: int = -1         # Sliding-Window-Größe (-1 = kein SW)

# protocols/engine.py — Protokoll-Definition COMMIT_STORE
"COMMIT_STORE": ProtocolDefinition(
    payload_classes=[KeyType, int, bytes],   # key, instance_id, cpu_data
    response_class=bool,
    handler_type=HandlerType.BLOCKING,
)
```

---

## Implementierung

### 1. `custom_types.py` — Payload um Per-Gruppe-Metadaten erweitern

Füge eine neue Struct hinzu (die alte bleibt für Abwärtskompatibilität erhalten):

```python
class GroupLayoutInfo(msgspec.Struct):
    """Layout-Metadaten für eine einzelne KV-Cache-Gruppe."""
    block_size: int
    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool
    tokens_per_block: int = 0   # aus EngineGroupInfo.tokens_per_block
```

Erweitere `RegisterEngineDrivenContextPayload` mit einem optionalen Feld:

```python
class RegisterEngineDrivenContextPayload(msgspec.Struct):
    instance_id: int
    model_name: str
    world_size: int
    block_size: int
    num_layers: int
    hidden_dim_size: int
    dtype_str: str
    use_mla: bool
    # NEU: bei Hybrid-Modellen gefüllt, bei Einzel-Gruppe leer
    group_layouts: list[GroupLayoutInfo] = []
```

Wenn `group_layouts` leer ist → Einzel-Gruppe (rückwärtskompatibel).
Wenn `group_layouts` N Einträge hat → N-Gruppen-Hybrid-Modell.

### 2. `transfer_context/base.py` — EngineDrivenContextMetadata erweitern

```python
@dataclass
class EngineDrivenContextMetadata:
    layout_desc: MemoryLayoutDesc       # Einzel-Gruppe (rückwärtskompatibel)
    block_size: int
    use_mla: bool
    # NEU: bei Hybrid-Modellen gefüllt
    group_layout_descs: list[MemoryLayoutDesc] = field(default_factory=list)
    group_block_sizes: list[int] = field(default_factory=list)
    group_use_mla: list[bool] = field(default_factory=list)
    group_blocks_in_chunk: list[int] = field(default_factory=list)

    @property
    def is_multi_group(self) -> bool:
        return len(self.group_layout_descs) > 1
```

Füge eine neue Hilfsfunktion hinzu:

```python
def slice_kv_caches_for_group(
    kv_caches: dict[str, torch.Tensor],
    layer_indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Extrahiere die Teilmenge der KV-Tensoren für eine Gruppe.

    Args:
        kv_caches: Alle Layer, geordnet wie vom Adapter übergeben.
        layer_indices: Indizes (0-basiert) der Schichten dieser Gruppe.

    Returns:
        Geordnetes Dict nur mit den Schichten dieser Gruppe.
    """
    all_values = list(kv_caches.values())
    return {str(i): all_values[idx] for i, idx in enumerate(sorted(layer_indices))}
```

Füge eine Wrapper-Funktion für Multi-Gruppe hinzu:

```python
def gather_paged_kv_multi_group_to_cpu(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],          # [group_idx][block_idx]
    engine_group_infos: list[EngineGroupInfo],
    lmcache_tokens_per_chunk: int,
    layout_hints: LayoutHints | None = None,
) -> list[list[torch.Tensor]]:
    """Gather alle KV-Gruppen zu CPU-Tensoren.

    Args:
        kv_caches: Alle Layer, nach Position sortiert.
        block_ids: Pro Gruppe eine Liste von Block-IDs.
        engine_group_infos: Gruppen-Metadaten aus der Registrierung.
        lmcache_tokens_per_chunk: LMCache Chunk-Größe in Tokens.
        layout_hints: Optional Layout-Hinweise.

    Returns:
        list[list[torch.Tensor]]: group_chunks[group_idx][chunk_idx]
    """
    import torch
    result = []
    for group_idx, group_info in enumerate(engine_group_infos):
        group_kv = slice_kv_caches_for_group(kv_caches, group_info.layer_indices)
        tokens_per_block = group_info.tokens_per_block
        if tokens_per_block <= 0:
            # Fallback: aus Tensor-Metadaten ermitteln
            block_size, _, _, _, engine_kv_format = compute_kv_layout(
                group_kv, layout_hints=layout_hints
            )
            tokens_per_block = block_size
        blocks_in_chunk = lmcache_tokens_per_chunk // tokens_per_block
        group_chunks = gather_paged_kv_to_cpu(
            group_kv,
            block_ids[group_idx],
            blocks_in_chunk,
            layout_hints=layout_hints,
        )
        result.append(group_chunks)
    return result


def scatter_cpu_multi_group_to_paged_kv(
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    group_chunks: list[list[torch.Tensor]],
    engine_group_infos: list[EngineGroupInfo],
    lmcache_tokens_per_chunk: int,
    skip_first_n_tokens: int = 0,
    layout_hints: LayoutHints | None = None,
) -> None:
    """Scatter CPU-Tensoren zurück in GPU-Paged-KV-Cache für alle Gruppen.

    Args:
        kv_caches: Alle Layer, nach Position sortiert.
        block_ids: Pro Gruppe eine Liste von Block-IDs.
        group_chunks: group_chunks[group_idx][chunk_idx] = CPU-Tensor.
        engine_group_infos: Gruppen-Metadaten aus der Registrierung.
        lmcache_tokens_per_chunk: LMCache Chunk-Größe in Tokens.
        skip_first_n_tokens: Tokens am Anfang überspringen (für APC).
        layout_hints: Optional Layout-Hinweise.
    """
    for group_idx, group_info in enumerate(engine_group_infos):
        group_kv = slice_kv_caches_for_group(kv_caches, group_info.layer_indices)
        tokens_per_block = group_info.tokens_per_block
        if tokens_per_block <= 0:
            block_size, _, _, _, _ = compute_kv_layout(group_kv, layout_hints=layout_hints)
            tokens_per_block = block_size
        blocks_in_chunk = lmcache_tokens_per_chunk // tokens_per_block
        scatter_cpu_to_paged_kv(
            group_kv,
            block_ids[group_idx],
            group_chunks[group_idx],
            blocks_in_chunk,
            skip_first_n_tokens=skip_first_n_tokens if group_idx == 0 else 0,
            layout_hints=layout_hints,
        )
```

### 3. `transfer_context/worker_transfer.py` — EngineDrivenTransferContext

#### 3a) Registrierung

```python
class EngineDrivenTransferContext(TransferContext):

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        # NEU
        self._engine_group_infos: list[EngineGroupInfo] = []
        self._lmcache_tokens_per_chunk: int = 256

    def register(
        self,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        model_name: str,
        world_size: int,
        blocks_in_chunk: int,
        mq_client: MessageQueueClient,
        mq_timeout: float,
        send_request: SendRequest,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: Sequence[EngineGroupInfo] = (),
    ) -> None:
        self._layout_hints = layout_hints
        self._engine_group_infos = list(engine_group_infos)
        num_groups = len(engine_group_infos)
        is_multi_group = num_groups > 1

        if not is_multi_group:
            # --- bestehender Einzel-Gruppen-Pfad (unverändert) ---
            (block_size, num_layers, hidden_dim_size,
             dtype_str, engine_kv_format) = compute_kv_layout(kv_caches, layout_hints=layout_hints)
            self._engine_kv_format = engine_kv_format
            use_mla_flag = is_mla(engine_kv_format)
            # chunks_size = blocks_in_chunk * block_size
            shape = (
                torch.Size([num_layers, blocks_in_chunk * block_size, hidden_dim_size])
                if use_mla_flag
                else torch.Size([2, num_layers, blocks_in_chunk * block_size, hidden_dim_size])
            )
            dtype = getattr(torch, dtype_str)
            layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
            group_layouts: list[GroupLayoutInfo] = []
        else:
            # --- NEU: Multi-Gruppen-Pfad ---
            kv_list = list(kv_caches.values())
            group_layouts_list: list[GroupLayoutInfo] = []
            group_layout_descs: list[MemoryLayoutDesc] = []
            group_block_sizes: list[int] = []
            group_use_mla_flags: list[bool] = []
            group_blocks_in_chunk: list[int] = []
            combined_layout_shapes: list[torch.Size] = []
            combined_layout_dtypes: list[torch.dtype] = []

            for group_info in engine_group_infos:
                group_kv = slice_kv_caches_for_group(kv_caches, group_info.layer_indices)
                (g_block_size, g_num_layers, g_hidden_dim,
                 g_dtype_str, g_fmt) = compute_kv_layout(group_kv, layout_hints=layout_hints)

                # tokens_per_block aus EngineGroupInfo nehmen wenn vorhanden,
                # sonst aus Tensor-Metadaten
                tpb = group_info.tokens_per_block if group_info.tokens_per_block > 0 else g_block_size
                g_blocks_in_chunk = lmcache_tokens_per_chunk // tpb  # lmcache_tokens_per_chunk = blocks_in_chunk * (tokens aus Einzel-Gruppe)

                # HINWEIS: lmcache_tokens_per_chunk muss aus blocks_in_chunk rekonstruiert werden
                # blocks_in_chunk ist der Wert für Gruppe 0 (oder alle Gruppen bei Einzel-Gruppe)
                # Bei Multi-Gruppe: jede Gruppe hat eigene blocks_in_chunk
                # Dieser Wert wird im Aufrufer (vllm_multi_process_adapter) pro Gruppe gesetzt
                # → Übergabe über engine_group_infos.tokens_per_block

                g_use_mla = is_mla(g_fmt)
                g_dtype = getattr(torch, g_dtype_str)
                g_chunk_tokens = g_blocks_in_chunk * tpb
                g_shape = (
                    torch.Size([g_num_layers, g_chunk_tokens, g_hidden_dim])
                    if g_use_mla
                    else torch.Size([2, g_num_layers, g_chunk_tokens, g_hidden_dim])
                )
                g_layout_desc = MemoryLayoutDesc(shapes=[g_shape], dtypes=[g_dtype])
                group_layout_descs.append(g_layout_desc)
                group_block_sizes.append(g_block_size)
                group_use_mla_flags.append(g_use_mla)
                group_blocks_in_chunk.append(g_blocks_in_chunk)
                combined_layout_shapes.append(g_shape)
                combined_layout_dtypes.append(g_dtype)
                group_layouts_list.append(GroupLayoutInfo(
                    block_size=g_block_size,
                    num_layers=g_num_layers,
                    hidden_dim_size=g_hidden_dim,
                    dtype_str=g_dtype_str,
                    use_mla=g_use_mla,
                    tokens_per_block=tpb,
                ))

            # Layout-Desc für die Server-Registry: alle Gruppen zusammengefasst
            # (der Server braucht pro Gruppe einen eigenen MemoryLayoutDesc für reserve_write)
            # → erster Eintrag als "default" für die Registry
            layout_desc = group_layout_descs[0]
            group_layouts = group_layouts_list

            # Für Einzel-Gruppe-Felder: Werte der ersten Gruppe
            block_size = group_block_sizes[0]
            dtype_str = group_layouts_list[0].dtype_str
            num_layers = group_layouts_list[0].num_layers
            hidden_dim_size = group_layouts_list[0].hidden_dim_size
            use_mla_flag = group_use_mla_flags[0]

        # Registrierungsnachricht senden
        future = send_request(
            mq_client,
            RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
            [
                RegisterEngineDrivenContextPayload(
                    instance_id=instance_id,
                    model_name=model_name,
                    world_size=world_size,
                    block_size=block_size,
                    num_layers=num_layers,
                    hidden_dim_size=hidden_dim_size,
                    dtype_str=dtype_str,
                    use_mla=use_mla_flag,
                    group_layouts=group_layouts,   # leer = Einzel-Gruppe
                )
            ],
        )
        response = future.result(timeout=mq_timeout)

        # Lokales Context-Objekt aufbauen
        if is_multi_group:
            metadata = EngineDrivenContextMetadata(
                layout_desc=layout_desc,
                block_size=block_size,
                use_mla=use_mla_flag,
                group_layout_descs=group_layout_descs,
                group_block_sizes=group_block_sizes,
                group_use_mla=group_use_mla_flags,
                group_blocks_in_chunk=group_blocks_in_chunk,
            )
        else:
            metadata = EngineDrivenContextMetadata(
                layout_desc=layout_desc,
                block_size=block_size,
                use_mla=use_mla_flag,
            )

        shm_name = ""
        pool_size = 0
        if isinstance(response, RegisterEngineDrivenContextResponse):
            shm_name = response.shm_name
            pool_size = response.pool_size
        self._engine_driven_context = create_engine_driven_context(
            metadata, mq_client, mq_timeout, shm_name=shm_name, pool_size=pool_size
        )
```

#### 3b) submit_store — Multi-Gruppen-Gather

```python
    def submit_store(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError("Context not registered. Call register() first.")

        torch_dev.synchronize()

        if len(self._engine_group_infos) > 1:
            # --- Multi-Gruppen-Pfad ---
            # Gather aller Gruppen zu CPU
            group_chunks = gather_paged_kv_multi_group_to_cpu(
                kv_caches,
                block_ids,
                self._engine_group_infos,
                lmcache_tokens_per_chunk=blocks_in_chunk * self._engine_driven_context.metadata.block_size,
                layout_hints=self._layout_hints,
            )
            # Serialisierung: list[list[Tensor]] → bytes
            # Format: pickle.dumps([(chunk0_np, chunk1_np, ...), ...])
            # = eine Liste pro Gruppe, jede Gruppe hat eine Liste von numpy-Arrays
            cpu_data = _serialize_multi_group_chunks(group_chunks)
            ok = self._engine_driven_context.commit_store(key, instance_id, _MULTI_GROUP_SENTINEL)
            # Direkter Weg: Bypass des EngineDrivenContext für multi-group
            # (commit_store der EngineDrivenContext kennt kein multi-group)
            # → besser: neues commit_store_multi_group
            # (Details: siehe Punkt 4)
        else:
            # --- bestehender Einzel-Gruppen-Pfad ---
            result = self._engine_driven_context.prepare_store(key, instance_id)
            out_buffers, chunk_indices = result if result is not None else (None, None)
            if chunk_indices is not None and len(chunk_indices) == 0:
                future: MessagingFuture[bool] = MessagingFuture()
                future.set_result(True)
                return future
            cpu_chunks = gather_paged_kv_to_cpu(
                kv_caches,
                block_ids[0],
                blocks_in_chunk,
                layout_hints=self._layout_hints,
                engine_kv_format=self._engine_kv_format,
                out=out_buffers,
                chunk_indices=chunk_indices,
            )
            if out_buffers is not None:
                torch_dev.synchronize()
            ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

        future = MessagingFuture()
        future.set_result(ok)
        return future
```

**Serialisierungsformat für Multi-Gruppe:**

```python
import pickle

def _serialize_multi_group_chunks(
    group_chunks: list[list[torch.Tensor]],
) -> bytes:
    """Serialisiere mehrere Gruppen als einen kompakten Bytes-Blob.

    Format: pickle.dumps([
        [numpy_array_chunk0, numpy_array_chunk1, ...],  # Gruppe 0
        [numpy_array_chunk0, numpy_array_chunk1, ...],  # Gruppe 1
        ...
    ])

    numpy-Arrays statt torch.Tensors wählen, um torch-Abhängigkeit
    im Server-Deserialisierungspfad zu minimieren und um pickle-Sicherheit
    zu gewährleisten.
    """
    serializable = [
        [chunk.contiguous().numpy() for chunk in group]
        for group in group_chunks
    ]
    return pickle.dumps(serializable)


def _deserialize_multi_group_chunks(
    cpu_data: bytes,
) -> list[list[torch.Tensor]]:
    """Deserialisiere Multi-Gruppen-Blob zurück zu Tensor-Listen."""
    raw = pickle.loads(cpu_data)
    return [
        [torch.from_numpy(arr) for arr in group]
        for group in raw
    ]
```

#### 3c) submit_retrieve — Multi-Gruppen-Scatter

```python
    def submit_retrieve(
        self,
        _request_id: str,
        key: Any,
        instance_id: int,
        kv_caches: dict[str, torch.Tensor],
        block_ids: list[list[int]],
        _event: IPCEvent,
        blocks_in_chunk: int,
        skip_first_n_tokens: int = 0,
    ) -> MessagingFuture:
        if self._engine_driven_context is None:
            raise RuntimeError("Context not registered.")

        if len(self._engine_group_infos) > 1:
            # --- Multi-Gruppen-Pfad ---
            src_buffers = self._engine_driven_context.prepare_retrieve_multi_group(key, instance_id)
            ok = src_buffers is not None
            if src_buffers is not None:
                group_chunks = _deserialize_multi_group_chunks(src_buffers)
                scatter_cpu_multi_group_to_paged_kv(
                    kv_caches,
                    block_ids,
                    group_chunks,
                    self._engine_group_infos,
                    lmcache_tokens_per_chunk=blocks_in_chunk * self._engine_driven_context.metadata.block_size,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                )
                torch_dev.synchronize()
            self._engine_driven_context.commit_retrieve(key, instance_id)
        else:
            # bestehender Einzel-Gruppen-Pfad (unverändert)
            src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
            ok = src_buffers is not None
            if src_buffers is not None:
                scatter_cpu_to_paged_kv(
                    kv_caches, block_ids[0], src_buffers, blocks_in_chunk,
                    skip_first_n_tokens=skip_first_n_tokens,
                    layout_hints=self._layout_hints,
                    engine_kv_format=self._engine_kv_format,
                )
                torch_dev.synchronize()
            self._engine_driven_context.commit_retrieve(key, instance_id)

        future: MessagingFuture[bool] = MessagingFuture()
        future.set_result(ok)
        return future
```

### 4. `modules/engine_driven_transfer.py` — Server-Seite

#### 4a) Registrierung mit Per-Gruppe-Layout

```python
def register_kv_cache_engine_driven_context(
    self,
    payload: RegisterEngineDrivenContextPayload,
) -> RegisterEngineDrivenContextResponse:
    is_multi_group = len(payload.group_layouts) > 1

    if not is_multi_group:
        # --- bestehender Pfad (rückwärtskompatibel) ---
        dtype = _get_dtype(payload.dtype_str)
        shape = _make_chunk_shape(
            payload.use_mla, payload.num_layers,
            self._ctx.chunk_size, payload.hidden_dim_size,
        )
        layout_desc = MemoryLayoutDesc(shapes=[shape], dtypes=[dtype])
        metadata = EngineDrivenContextMetadata(
            layout_desc=layout_desc,
            block_size=payload.block_size,
            use_mla=payload.use_mla,
        )
        self._ctx.layout_desc_registry.register(
            payload.model_name, payload.world_size, layout_desc
        )
    else:
        # --- NEU: Multi-Gruppen-Registrierung ---
        group_layout_descs: list[MemoryLayoutDesc] = []
        group_block_sizes: list[int] = []
        group_use_mla: list[bool] = []
        group_blocks_in_chunk: list[int] = []

        for g in payload.group_layouts:
            g_dtype = _get_dtype(g.dtype_str)
            tpb = g.tokens_per_block if g.tokens_per_block > 0 else g.block_size
            g_blocks_in_chunk = self._ctx.chunk_size // tpb
            g_chunk_tokens = g_blocks_in_chunk * tpb
            g_shape = _make_chunk_shape(g.use_mla, g.num_layers, g_chunk_tokens, g.hidden_dim_size)
            g_layout_desc = MemoryLayoutDesc(shapes=[g_shape], dtypes=[g_dtype])
            group_layout_descs.append(g_layout_desc)
            group_block_sizes.append(g.block_size)
            group_use_mla.append(g.use_mla)
            group_blocks_in_chunk.append(g_blocks_in_chunk)

        metadata = EngineDrivenContextMetadata(
            layout_desc=group_layout_descs[0],
            block_size=group_block_sizes[0],
            use_mla=group_use_mla[0],
            group_layout_descs=group_layout_descs,
            group_block_sizes=group_block_sizes,
            group_use_mla=group_use_mla,
            group_blocks_in_chunk=group_blocks_in_chunk,
        )
        # Für Registry: nur Gruppe 0 (wird für Lookups genutzt)
        self._ctx.layout_desc_registry.register(
            payload.model_name, payload.world_size, group_layout_descs[0]
        )

    self._engine_driven_contexts[payload.instance_id] = EngineDrivenContextEntry(
        metadata=metadata,
        model_name=payload.model_name,
        world_size=payload.world_size,
    )
    # TransferStrategy erzeugen (wie bisher)
    ...
```

#### 4b) commit_store — Multi-Gruppen-Speicherung

```python
def commit_store(
    self,
    key: IPCCacheServerKey,
    instance_id: int,
    cpu_data: bytes,
) -> bool:
    entry = self._engine_driven_contexts.get(instance_id)
    if entry is None:
        raise ValueError(f"No context for instance {instance_id}")

    if entry.metadata.is_multi_group:
        return self._commit_store_multi_group(key, instance_id, cpu_data, entry)
    else:
        # bestehender Pfad (unverändert)
        strategy = self._strategies[instance_id]
        return strategy.commit_store(
            key=key, instance_id=instance_id, cpu_data=cpu_data,
            context=entry.metadata,
            resolve_obj_keys=lambda k: self._resolve_single_group_obj_keys(k),
        )

def _commit_store_multi_group(
    self,
    key: IPCCacheServerKey,
    instance_id: int,
    cpu_data: bytes,
    entry: EngineDrivenContextEntry,
) -> bool:
    """Speichere multi-group Daten in alle Object Groups."""
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        _deserialize_multi_group_chunks,
    )

    group_chunks_all = _deserialize_multi_group_chunks(cpu_data)
    num_groups = len(entry.metadata.group_layout_descs)
    assert len(group_chunks_all) == num_groups, (
        f"Expected {num_groups} groups, got {len(group_chunks_all)}"
    )

    # Object-Keys für alle Gruppen auflösen
    obj_keys_per_group = self._ctx.resolve_obj_keys(key, list(range(num_groups)))

    all_reserved: dict[ObjectKey, MemoryObj] = {}
    try:
        for group_idx in range(num_groups):
            obj_keys = obj_keys_per_group[group_idx]
            layout_desc = entry.metadata.group_layout_descs[group_idx]
            reserved = self._ctx.storage_manager.reserve_write(obj_keys, layout_desc, "new")
            all_reserved.update(reserved)

            group_chunks = group_chunks_all[group_idx]
            for obj_key, chunk_tensor in zip(obj_keys, group_chunks):
                if obj_key not in reserved:
                    continue   # bereits gecacht
                memory_obj = reserved[obj_key]
                chunk_bytes = chunk_tensor.contiguous().numpy().tobytes()
                memory_obj.raw_data[:len(chunk_bytes)] = chunk_bytes

        self._ctx.storage_manager.finish_write(list(all_reserved.keys()))
        return True
    except Exception:
        logger.exception("commit_store_multi_group failed")
        return False
```

#### 4c) prepare_retrieve / commit_retrieve — Multi-Gruppen-Laden

```python
def prepare_retrieve(
    self,
    key: IPCCacheServerKey,
    instance_id: int,
) -> PrepareRetrieveResponse:
    entry = self._engine_driven_contexts.get(instance_id)
    if entry.metadata.is_multi_group:
        return self._prepare_retrieve_multi_group(key, instance_id, entry)
    else:
        # bestehender Pfad
        ...

def _prepare_retrieve_multi_group(
    self,
    key: IPCCacheServerKey,
    instance_id: int,
    entry: EngineDrivenContextEntry,
) -> PrepareRetrieveResponse:
    """Lade alle Gruppen und serialisiere als Multi-Group-Blob."""
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        _serialize_multi_group_chunks,
    )
    import torch

    num_groups = len(entry.metadata.group_layout_descs)
    obj_keys_per_group = self._ctx.resolve_obj_keys(key, list(range(num_groups)))

    group_chunks_all: list[list[torch.Tensor]] = []
    prefetched_keys: list[ObjectKey] = []

    try:
        for group_idx in range(num_groups):
            obj_keys = obj_keys_per_group[group_idx]
            with self._ctx.storage_manager.read_prefetched_results(obj_keys) as memory_objs:
                if not memory_objs or len(memory_objs) != len(obj_keys):
                    return PrepareRetrieveResponse(cpu_data=b"")  # Cache-Miss

                chunks = []
                layout_desc = entry.metadata.group_layout_descs[group_idx]
                for mem_obj in memory_objs:
                    shape = layout_desc.shapes[0]
                    dtype = layout_desc.dtypes[0]
                    tensor = torch.frombuffer(
                        mem_obj.raw_data[:mem_obj.get_size()], dtype=dtype
                    ).view(shape).clone()
                    chunks.append(tensor)
                group_chunks_all.append(chunks)
                prefetched_keys.extend(obj_keys)

        cpu_data = _serialize_multi_group_chunks(group_chunks_all)
        # Prefetch-Locks erst nach Serialisierung freigeben
        self._ctx.storage_manager.finish_read_prefetched(prefetched_keys)
        return PrepareRetrieveResponse(cpu_data=cpu_data)
    except Exception:
        logger.exception("prepare_retrieve_multi_group failed")
        return PrepareRetrieveResponse(cpu_data=b"")
```

### 5. `server_transfer.py` — Minimale Anpassung

Keine inhaltliche Änderung erforderlich. Die `commit_store`-Schnittstelle nimmt `bytes` entgegen
und leitet sie durch. Da Multi-Gruppe direkt in `engine_driven_transfer.py` behandelt wird
(ohne `TransferStrategy`), müssen hier keine Änderungen vorgenommen werden. Optional: Assertion
hinzufügen, dass `is_multi_group=True`-Aufrufe nie in `PickleTransferStrategy.commit_store` landen.

---

## Protokollrelevante Anpassungen

### PrepareRetrieveResponse erweitern

Das bestehende `PrepareRetrieveResponse` in `protocols/engine.py` muss `cpu_data: bytes`
enthalten können, wenn Multi-Gruppe aktiv ist. Prüfe ob die Antwortklasse das bereits hat oder
ergänze:

```python
class PrepareRetrieveResponse(msgspec.Struct):
    """Response für PREPARE_RETRIEVE."""
    cpu_data: bytes = b""   # Leer = SHM-Pfad / Cache-Miss; gefüllt = Pickle-Pfad
    # bestehende SHM-Felder bleiben erhalten
    ...
```

---

## Aktivierung (kein VRAM im Server)

Nach der Implementierung wird der engine-driven Pfad für Hybrid-Modelle durch eine der folgenden
Methoden aktiviert:

**Option A: Umgebungsvariable (einfachste Methode)**
```bash
export LMCACHE_MP_TRANSFER_MODE=engine_driven
lmcache server --supported-transfer-mode auto ...
vllm serve Qwen/Qwen3.6-27B ...
```

**Option B: kv_connector_extra_config in vLLM**
```json
{
  "kv_connector": "LMCacheMPConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "mp_transfer_mode": "engine_driven"
  }
}
```

Der Server muss mit `--supported-transfer-mode engine_driven` oder `auto` gestartet werden.
`auto` ist empfohlen, weil es beide Pfade gleichzeitig unterstützt (für Mischbetrieb mit
nicht-hybriden Modellen).

---

## Testanforderungen

Schreibe Unit-Tests unter `tests/v1/multiprocess/` (oder wo bestehende Tests liegen):

### Test 1: `test_serialize_deserialize_multi_group`
```python
def test_serialize_deserialize_multi_group():
    """Roundtrip: group_chunks → bytes → group_chunks"""
    from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
        _serialize_multi_group_chunks, _deserialize_multi_group_chunks
    )
    group_chunks = [
        [torch.randn(2, 4, 16, 128), torch.randn(2, 4, 16, 128)],  # Gruppe 0: 2 Chunks
        [torch.randn(1, 2, 784, 64)],                               # Gruppe 1: 1 Chunk (GDN)
    ]
    blob = _serialize_multi_group_chunks(group_chunks)
    assert isinstance(blob, bytes)
    restored = _deserialize_multi_group_chunks(blob)
    assert len(restored) == 2
    assert len(restored[0]) == 2
    assert len(restored[1]) == 1
    torch.testing.assert_close(restored[0][0], group_chunks[0][0])
    torch.testing.assert_close(restored[1][0], group_chunks[1][0])
```

### Test 2: `test_slice_kv_caches_for_group`
```python
def test_slice_kv_caches_for_group():
    from lmcache.v1.multiprocess.transfer_context.base import slice_kv_caches_for_group
    kv = {str(i): torch.zeros(2, 100, 64) for i in range(10)}
    group = EngineGroupInfo(engine_group_id=0, layer_indices=(0, 3, 7))
    sliced = slice_kv_caches_for_group(kv, group.layer_indices)
    assert len(sliced) == 3
    # Reihenfolge ist sortiert nach layer_indices
    for key, tensor in sliced.items():
        assert tensor.shape == (2, 100, 64)
```

### Test 3: `test_engine_driven_multi_group_register_payload`
```python
def test_group_layout_in_payload():
    from lmcache.v1.multiprocess.custom_types import (
        RegisterEngineDrivenContextPayload, GroupLayoutInfo
    )
    g0 = GroupLayoutInfo(block_size=16, num_layers=14, hidden_dim_size=256,
                          dtype_str="bfloat16", use_mla=False, tokens_per_block=16)
    g1 = GroupLayoutInfo(block_size=784, num_layers=14, hidden_dim_size=512,
                          dtype_str="float32", use_mla=False, tokens_per_block=784)
    payload = RegisterEngineDrivenContextPayload(
        instance_id=1, model_name="test", world_size=2,
        block_size=16, num_layers=14, hidden_dim_size=256,
        dtype_str="bfloat16", use_mla=False,
        group_layouts=[g0, g1],
    )
    # msgspec Roundtrip
    import msgspec.msgpack as mp
    raw = mp.encode(payload)
    restored = mp.decode(raw, type=RegisterEngineDrivenContextPayload)
    assert len(restored.group_layouts) == 2
    assert restored.group_layouts[1].tokens_per_block == 784
```

### Test 4: `test_engine_driven_multi_group_store_retrieve` (Integration)
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Braucht GPU für gather")
def test_engine_driven_multi_group_store_retrieve():
    """End-to-End: Multi-Gruppe gather → serialize → deserialize → scatter"""
    from lmcache.v1.multiprocess.transfer_context.base import (
        gather_paged_kv_multi_group_to_cpu,
        scatter_cpu_multi_group_to_paged_kv,
    )
    # Simuliere Qwen3.6-27B Struktur: 28 Attention-Layer (Gruppe 0) + 28 GDN-Layer (Gruppe 1)
    # Vereinfacht: 2 Layer pro Gruppe, block_size=16 für Attn, block_size=784 für GDN
    attn_kv = {str(i): torch.randn(2, 100, 16, 128, device="cuda") for i in range(2)}   # [kv, blocks, block_size, head_dim]
    gdn_kv = {str(i): torch.randn(1, 100, 784, 64, device="cuda") for i in range(2)}    # [blocks, state_size, ...]
    all_kv = {**{f"attn_{k}": v for k, v in attn_kv.items()},
              **{f"gdn_{k}": v for k, v in gdn_kv.items()}}
    # ... (vollständige Implementierung je nach konkretem Tensor-Format)
```

### Test 5: `test_single_group_backward_compat`
```python
def test_single_group_still_works():
    """Einzel-Gruppe-Modelle (Llama, Qwen3-27B dense) dürfen nicht regressieren."""
    payload = RegisterEngineDrivenContextPayload(
        instance_id=1, model_name="test", world_size=1,
        block_size=16, num_layers=32, hidden_dim_size=256,
        dtype_str="float16", use_mla=False,
        group_layouts=[],  # leer = Einzel-Gruppe
    )
    assert len(payload.group_layouts) == 0
```

---

## Wichtige Implementierungshinweise

### lmcache_tokens_per_chunk im EngineDrivenTransferContext verfügbar machen

`blocks_in_chunk` ist der einzige Parameter, der `lmcache_tokens_per_chunk` kodiert (für
Einzel-Gruppe: `lmcache_tokens_per_chunk = blocks_in_chunk × block_size`). Bei Multi-Gruppe
hat jede Gruppe einen anderen `block_size`. Speichere daher `lmcache_tokens_per_chunk` explizit
im Context oder leite es beim Store/Retrieve von `blocks_in_chunk × block_size[0]` ab.

### GDN-State-Sonderfall (tokens_per_block = chunk_size)

Für GDN-Layer in Qwen3.6-27B gilt: ein "Block" = ein vollständiger LMCache-Chunk (784 Tokens).
Das bedeutet:
- `tokens_per_block = 784`
- `blocks_in_chunk = 784 / 784 = 1`
- `gather_paged_kv_to_cpu` wird mit `blocks_in_chunk=1` und einer einzelnen Block-ID pro Chunk
  aufgerufen — das reduziert sich auf eine einfache Tensor-Kopie.

Diese Invariante muss in Tests explizit geprüft werden.

### Keine Änderung an `lmcache_mp_connector.py` (vLLM-Integration)

Der bestehende Connector übergibt bereits `engine_group_infos` an den Adapter. Durch das Setzen
von `LMCACHE_MP_TRANSFER_MODE=engine_driven` wählt `create_transfer_context()` automatisch
`EngineDrivenTransferContext`. Der Connector selbst muss nicht geändert werden.

### Abwärtskompatibilität

- Alle bestehenden Einzel-Gruppen-Pfade bleiben **unverändert** und weiterhin aktiv.
- `group_layouts=[]` in `RegisterEngineDrivenContextPayload` bedeutet Einzel-Gruppe.
- Server-seitig prüft `entry.metadata.is_multi_group` an jedem Entscheidungspunkt.
- Die msgspec-Serialisierung mit `default=[]` ist abwärtskompatibel mit alten Clients.

### Fehlerbehandlung

- Wenn `len(block_ids) > 1` aber `len(engine_group_infos) == 0` (kein Hybrid angemeldet):
  → `RuntimeError` mit klarer Meldung, nicht mit dem alten generischen Text.
- Wenn Gruppen-Anzahl in `block_ids` und `engine_group_infos` nicht übereinstimmt:
  → `ValueError` mit Gruppen-Counts in der Meldung.

---

---

## Lücken und Korrekturen gegenüber dem obigen Skizzen-Code

Die obigen Codeblöcke sind **Skizzen mit Lücken**, die folgende Abschnitte schließen.

### 6. `transfer_context/base.py` — Neue Methoden an `EngineDrivenContext` ABC

`EngineDrivenContextPickle.commit_store` nimmt `list[torch.Tensor]` entgegen und serialisiert
sie selbst. Für Multi-Gruppe müssen wir vorab serialisierte `bytes` senden (der Caller hat
bereits `_serialize_multi_group_chunks` aufgerufen). Wir brauchen daher **zwei neue
nicht-abstrakte Methoden** in der Basisklasse, damit der MQ-Aufruf nicht im Aufrufer dupliziert
wird:

```python
class EngineDrivenContext(ABC):
    # ... bestehende abstrakte Methoden unverändert ...

    # NEU: nicht-abstrakte Convenience-Methoden für Multi-Gruppe
    # (Konkrete Unterklassen müssen diese NICHT überschreiben)

    def commit_store_raw(
        self, key: IPCCacheServerKey, instance_id: int, cpu_data: bytes
    ) -> bool:
        """Sende vorserialisierten Bytes-Blob direkt via COMMIT_STORE.

        Für Multi-Gruppe: Aufrufer hat bereits _serialize_multi_group_chunks()
        aufgerufen. Kein weiteres pickle.dumps() hier.
        Funktioniert für beide konkreten Unterklassen (Pickle und SHM),
        da der MQ-Aufruf in beiden Fällen gleich ist.
        """
        future = self.mq_client.submit_request(
            RequestType.COMMIT_STORE,
            [key, instance_id, cpu_data],
            get_response_class(RequestType.COMMIT_STORE),
        )
        try:
            return bool(future.result(timeout=self.mq_timeout))
        except TimeoutError:
            return False

    def prepare_retrieve_raw(
        self, key: IPCCacheServerKey, instance_id: int
    ) -> bytes | None:
        """Sende PREPARE_RETRIEVE und gib rohe Bytes zurück (kein pickle.loads).

        Für Multi-Gruppe: Aufrufer deserialisiert mit
        _deserialize_multi_group_chunks() selbst.
        Gibt None zurück bei Cache-Miss oder Timeout.
        """
        future = self.mq_client.submit_request(
            RequestType.PREPARE_RETRIEVE,
            [key, instance_id],
            get_response_class(RequestType.PREPARE_RETRIEVE),
        )
        try:
            response = future.result(timeout=self.mq_timeout)
        except TimeoutError:
            return None
        if not response.success or not response.data:
            return None
        return response.data   # bytes, nicht deserialisiertm
```

Mit diesen Methoden vereinfacht sich `submit_store`/`submit_retrieve` für Multi-Gruppe
erheblich (kein direkter MQ-Zugriff nötig, keine Sentinel-Werte).

### 7. `transfer_context/worker_transfer.py` — Korrigierter submit_store/retrieve

Ersetzt den Skizzen-Code aus Abschnitt 3b/3c vollständig:

```python
def submit_store(
    self,
    _request_id: str,
    key: Any,
    instance_id: int,
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    _event: IPCEvent,
    blocks_in_chunk: int,
) -> MessagingFuture:
    if self._engine_driven_context is None:
        raise RuntimeError("Context not registered. Call register() first.")

    torch_dev.synchronize()

    if len(self._engine_group_infos) > 1:
        # ── Multi-Gruppen-Pfad ────────────────────────────────────────────
        assert len(block_ids) == len(self._engine_group_infos), (
            f"block_ids hat {len(block_ids)} Gruppen, "
            f"aber {len(self._engine_group_infos)} engine_group_infos registriert"
        )
        group_chunks = gather_paged_kv_multi_group_to_cpu(
            kv_caches,
            block_ids,
            self._engine_group_infos,
            lmcache_tokens_per_chunk=self._lmcache_tokens_per_chunk,
            layout_hints=self._layout_hints,
        )
        cpu_data = _serialize_multi_group_chunks(group_chunks)
        # prepare_store wird auch für multi-group aufgerufen (Server-seitig
        # notwendig für Prefetch/Lock-Verwaltung), aber die SHM-Slots
        # werden ignoriert (keine Puffer für multi-group voralloziert).
        self._engine_driven_context.prepare_store(key, instance_id)
        ok = self._engine_driven_context.commit_store_raw(key, instance_id, cpu_data)
    else:
        # ── Einzel-Gruppen-Pfad (unverändert) ────────────────────────────
        result = self._engine_driven_context.prepare_store(key, instance_id)
        out_buffers, chunk_indices = result if result is not None else (None, None)
        if chunk_indices is not None and len(chunk_indices) == 0:
            future: MessagingFuture[bool] = MessagingFuture()
            future.set_result(True)
            return future
        cpu_chunks = gather_paged_kv_to_cpu(
            kv_caches,
            block_ids[0],
            blocks_in_chunk,
            layout_hints=self._layout_hints,
            engine_kv_format=self._engine_kv_format,
            out=out_buffers,
            chunk_indices=chunk_indices,
        )
        if out_buffers is not None:
            torch_dev.synchronize()
        ok = self._engine_driven_context.commit_store(key, instance_id, cpu_chunks)

    future = MessagingFuture()
    future.set_result(ok)
    return future


def submit_retrieve(
    self,
    _request_id: str,
    key: Any,
    instance_id: int,
    kv_caches: dict[str, torch.Tensor],
    block_ids: list[list[int]],
    _event: IPCEvent,
    blocks_in_chunk: int,
    skip_first_n_tokens: int = 0,
) -> MessagingFuture:
    if self._engine_driven_context is None:
        raise RuntimeError("Context not registered.")

    if len(self._engine_group_infos) > 1:
        # ── Multi-Gruppen-Pfad ────────────────────────────────────────────
        raw = self._engine_driven_context.prepare_retrieve_raw(key, instance_id)
        ok = raw is not None
        if raw:
            group_chunks = _deserialize_multi_group_chunks(raw)
            torch_dev.synchronize()
            scatter_cpu_multi_group_to_paged_kv(
                kv_caches,
                block_ids,
                group_chunks,
                self._engine_group_infos,
                lmcache_tokens_per_chunk=self._lmcache_tokens_per_chunk,
                skip_first_n_tokens=skip_first_n_tokens,
                layout_hints=self._layout_hints,
            )
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)
    else:
        # ── Einzel-Gruppen-Pfad (unverändert) ────────────────────────────
        src_buffers = self._engine_driven_context.prepare_retrieve(key, instance_id)
        ok = src_buffers is not None
        if src_buffers is not None:
            scatter_cpu_to_paged_kv(
                kv_caches, block_ids[0], src_buffers, blocks_in_chunk,
                skip_first_n_tokens=skip_first_n_tokens,
                layout_hints=self._layout_hints,
                engine_kv_format=self._engine_kv_format,
            )
            torch_dev.synchronize()
        self._engine_driven_context.commit_retrieve(key, instance_id)

    future: MessagingFuture[bool] = MessagingFuture()
    future.set_result(ok)
    return future
```

### 8. `_lmcache_tokens_per_chunk` korrekt berechnen und speichern

Der Wert `lmcache_tokens_per_chunk` ist der globale LMCache-Chunk-Parameter. Er ergibt sich aus
`blocks_in_chunk × attention_block_size` und muss einmalig in `register()` gespeichert werden:

```python
class EngineDrivenTransferContext(TransferContext):

    def __init__(self) -> None:
        self._engine_driven_context: EngineDrivenContext | None = None
        self._layout_hints: LayoutHints | None = None
        self._engine_kv_format: Any = None
        self._engine_group_infos: list[EngineGroupInfo] = []
        self._lmcache_tokens_per_chunk: int = 0   # ← NEU, wird in register() gesetzt

    def register(self, ..., blocks_in_chunk: int, ...) -> None:
        # Einzel-Gruppe:
        block_size, ... = compute_kv_layout(kv_caches, ...)
        self._lmcache_tokens_per_chunk = blocks_in_chunk * block_size

        # Multi-Gruppe (nach Berechnung von group_block_sizes[0]):
        self._lmcache_tokens_per_chunk = blocks_in_chunk * group_block_sizes[0]
        # Invariante: Alle Gruppen teilen denselben lmcache_tokens_per_chunk.
        # group_blocks_in_chunk[i] = lmcache_tokens_per_chunk // tpb[i]
```

**Invariante**: `lmcache_tokens_per_chunk` ist für alle Gruppen **gleich** (es ist ein globaler
LMCache-Parameter). Pro Gruppe variiert nur `blocks_in_chunk`:
```
Gruppe 0 (Attention, tpb=16):   blocks_in_chunk = 784 / 16 = 49
Gruppe 1 (GDN, tpb=784):        blocks_in_chunk = 784 / 784 = 1
```

### 9. SHM-Modus: Erzwungenes Fallback auf Pickle für Multi-Gruppe

Der SHM-Pool ist beim Server für eine feste `MemoryLayoutDesc`-Shape alloziert. Bei Multi-Gruppe
haben die Gruppen unterschiedliche Shapes, und der SHM-Pool ist nicht dafür dimensioniert.

In `create_engine_driven_context` (Datei `transfer_context/base.py`) muss Multi-Gruppe
explizit auf Pickle gezwungen werden:

```python
def create_engine_driven_context(
    metadata: EngineDrivenContextMetadata,
    mq_client: MessageQueueClient,
    mq_timeout: float,
    shm_name: str = "",
    pool_size: int = 0,
    use_pickle: bool = False,
) -> EngineDrivenContext:
    # NEU: Multi-Gruppe erzwingt Pickle (SHM-Pool ist single-group-sizing)
    if metadata.is_multi_group:
        use_pickle = True
        logger.info(
            "Multi-group engine-driven context: forcing pickle transport "
            "(SHM pool is sized for single-group layout)"
        )

    if not shm_name or pool_size <= 0:
        use_pickle = True

    if not use_pickle:
        # ... bestehender SHM-Pfad ...
```

Auf der **Server-Seite** muss `register_kv_cache_engine_driven_context` für Multi-Gruppe
**keine SHM-Name** in der Response zurückgeben (leerer String erzwingt Pickle auf Worker-Seite):

```python
def register_kv_cache_engine_driven_context(self, payload):
    if len(payload.group_layouts) > 1:
        # Kein SHM für Multi-Gruppe
        shm_name = ""
        pool_size = 0
    else:
        shm_name = self._shm_pool_info["shm_name"]
        pool_size = self._shm_pool_info["pool_size"]
    # ... Rest wie bisher ...
```

### 10. `create_transfer_context` — AUTO-Modus für Hybrid-Modelle

Ändere `create_transfer_context` in `worker_transfer.py`, damit im AUTO-Modus Hybrid-Modelle
**automatisch** auf `EngineDrivenTransferContext` umgestellt werden, ohne dass
`LMCACHE_MP_TRANSFER_MODE=engine_driven` gesetzt werden muss:

```python
def create_transfer_context(
    kv_caches: dict[str, torch.Tensor],
    mode: "str | MPTransferMode | None" = None,
    num_engine_groups: int = 1,   # ← NEU: 1 = standard, >1 = hybrid
) -> TransferContext:
    """...docstring ergänzen..."""
    if not kv_caches:
        raise ValueError("kv_caches is empty")
    device_types = {tensor.device.type for tensor in kv_caches.values()}
    if len(device_types) != 1:
        raise ValueError(f"All KV cache tensors must share one device type, got {device_types}")
    device_type = next(iter(device_types))
    resolved_mode = _resolve_mode(mode)

    logger.info(
        "Creating transfer context (device_type=%s, mode=%s, num_engine_groups=%d)",
        device_type, resolved_mode.value, num_engine_groups,
    )

    if resolved_mode is MPTransferMode.LMCACHE_DRIVEN:
        if num_engine_groups > 1:
            raise ValueError(
                "Transfer mode 'lmcache_driven' does not support hybrid models "
                "(num_engine_groups=%d). Use 'engine_driven' or 'auto'." % num_engine_groups
            )
        return _build_lmcache_driven_context(device_type)

    if resolved_mode is MPTransferMode.ENGINE_DRIVEN:
        return EngineDrivenTransferContext()

    # AUTO: bei CUDA und Einzel-Gruppe → lmcache-driven (bisheriges Verhalten)
    #       bei CUDA und Multi-Gruppe (hybrid) → engine-driven (kein VRAM im Server)
    if device_type == "cuda":
        if num_engine_groups > 1:
            logger.info(
                "AUTO mode: hybrid model (%d groups) detected → engine-driven "
                "(eliminates server-side VRAM)", num_engine_groups
            )
            return EngineDrivenTransferContext()
        return LMCacheDrivenTransferContext()

    return EngineDrivenTransferContext()
```

**Aufruf-Anpassung** in `vllm_multi_process_adapter.py` (Zeile 1139):

```python
# Alt:
transfer_ctx = create_transfer_context(kv_caches, mode=self._mp_transfer_mode)

# Neu:
transfer_ctx = create_transfer_context(
    kv_caches,
    mode=self._mp_transfer_mode,
    num_engine_groups=len(self.engine_group_infos) if self.engine_group_infos else 1,
)
```

`self.engine_group_infos` ist zu diesem Zeitpunkt bereits gesetzt (Zeile 1116 kommt vor 1139).

### 11. Server-seitiges commit_store: `_resolve_single_group_obj_keys` → `_resolve_multi_group_obj_keys`

Der bestehende Server-seitige Hilfsmethode `_resolve_single_group_obj_keys` löst nur
Object-Group-0 auf. Für Multi-Gruppe benötigen wir alle Groups:

```python
# Bestehend (bleibt für Einzel-Gruppe erhalten):
def _resolve_single_group_obj_keys(self, key: IPCCacheServerKey) -> list[ObjectKey]:
    return self._ctx.resolve_obj_keys(key, [0])[0]

# NEU:
def _resolve_all_group_obj_keys(
    self, key: IPCCacheServerKey, num_groups: int
) -> list[list[ObjectKey]]:
    """Löse Object-Keys für alle num_groups auf.
    Returns: obj_keys[group_idx] = list[ObjectKey]
    """
    return self._ctx.resolve_obj_keys(key, list(range(num_groups)))
```

### 12. Vollständige korrigierte `_commit_store_multi_group` Serverseite

```python
def _commit_store_multi_group(
    self,
    key: IPCCacheServerKey,
    instance_id: int,
    cpu_data: bytes,
    entry: EngineDrivenContextEntry,
) -> bool:
    from lmcache.v1.multiprocess.transfer_context.base import (
        _deserialize_multi_group_chunks,
    )

    group_chunks_all = _deserialize_multi_group_chunks(cpu_data)
    num_groups = len(entry.metadata.group_layout_descs)

    if len(group_chunks_all) != num_groups:
        logger.error(
            "commit_store_multi_group: Erwartete %d Gruppen, bekam %d",
            num_groups, len(group_chunks_all),
        )
        return False

    obj_keys_all = self._resolve_all_group_obj_keys(key, num_groups)
    all_reserved_keys: list[ObjectKey] = []

    try:
        for group_idx in range(num_groups):
            obj_keys = obj_keys_all[group_idx]
            layout_desc = entry.metadata.group_layout_descs[group_idx]
            group_chunks = group_chunks_all[group_idx]

            reserved = self._ctx.storage_manager.reserve_write(
                obj_keys, layout_desc, "new"
            )
            all_reserved_keys.extend(reserved.keys())

            for obj_key, chunk_tensor in zip(obj_keys, group_chunks):
                if obj_key not in reserved:
                    # Bereits gecacht (z. B. durch Prefix-Deduplication)
                    continue
                memory_obj = reserved[obj_key]
                chunk_np = chunk_tensor.contiguous().numpy()
                chunk_bytes = chunk_np.tobytes()
                nbytes = len(chunk_bytes)
                assert nbytes <= len(memory_obj.raw_data), (
                    f"Chunk {nbytes} Bytes > Puffer {len(memory_obj.raw_data)} Bytes "
                    f"(Gruppe {group_idx}, Shape {chunk_tensor.shape})"
                )
                memory_obj.raw_data[:nbytes] = chunk_bytes

        self._ctx.storage_manager.finish_write(all_reserved_keys)
        return True

    except Exception:
        logger.exception("_commit_store_multi_group fehlgeschlagen")
        if all_reserved_keys:
            # Halb-geschriebene Objekte freigeben
            self._ctx.storage_manager.finish_write(all_reserved_keys)
        return False
```

### 13. Server-seitiges `_prepare_retrieve_multi_group`

```python
def _prepare_retrieve_multi_group(
    self,
    key: IPCCacheServerKey,
    instance_id: int,
    entry: EngineDrivenContextEntry,
) -> PrepareRetrieveResponse:
    from lmcache.v1.multiprocess.transfer_context.base import (
        _serialize_multi_group_chunks,
    )
    import torch

    num_groups = len(entry.metadata.group_layout_descs)
    obj_keys_all = self._resolve_all_group_obj_keys(key, num_groups)

    group_chunks_all: list[list[torch.Tensor]] = []
    all_prefetched: list[ObjectKey] = []

    for group_idx in range(num_groups):
        obj_keys = obj_keys_all[group_idx]
        layout_desc = entry.metadata.group_layout_descs[group_idx]

        # Cache-Miss-Erkennung: wenn für irgendeinen Key kein Treffer → gesamtes Miss
        result = self._ctx.storage_manager.try_read_prefetched(obj_keys)
        if result is None:
            # Cache-Miss für diese Gruppe → auch alle schon gelesenen freigeben
            if all_prefetched:
                self._ctx.storage_manager.finish_read_prefetched(all_prefetched)
            return PrepareRetrieveResponse(success=False, data=b"")

        memory_objs, read_obj_keys = result
        all_prefetched.extend(read_obj_keys)

        shape = layout_desc.shapes[0]
        dtype = layout_desc.dtypes[0]

        chunks: list[torch.Tensor] = []
        for mem_obj in memory_objs:
            size = mem_obj.get_size()
            tensor = torch.frombuffer(
                bytes(mem_obj.raw_data[:size]),
                dtype=dtype,
            ).view(shape).clone()   # clone() → eigener Speicher, unabhängig vom SHM-Lock
            chunks.append(tensor)

        group_chunks_all.append(chunks)

    # Serialisieren
    cpu_data = _serialize_multi_group_chunks(group_chunks_all)

    # Read-Locks freigeben
    self._ctx.storage_manager.finish_read_prefetched(all_prefetched)

    return PrepareRetrieveResponse(success=True, data=cpu_data)
```

**Hinweis zu `try_read_prefetched`**: Die Methode `storage_manager.read_prefetched_results`
(oder wie sie in der tatsächlichen Implementierung heißt) muss für alle `obj_keys` einer Gruppe
erfolgreich sein, bevor die Daten als vorhanden gelten. Prüfe die tatsächliche API des
`StorageManager` und passe den Methodennamen entsprechend an. Das Prinzip ist:
- Alle Objekte einer Gruppe müssen im Cache sein (kein partieller Hit)
- Bei einem einzigen Miss: `success=False` für die gesamte Anfrage

### 14. `prepare_store` Serverseite für Multi-Gruppe

Da Multi-Gruppe immer Pickle-Modus nutzt (keine SHM-Slots), ist `prepare_store` für
Multi-Gruppe ein vollständiges No-op — nur Lock-Initialisierung:

```python
def prepare_store(self, key, instance_id) -> PrepareStoreResponse:
    entry = self._engine_driven_contexts.get(instance_id)
    if entry is None:
        raise ValueError(...)

    if entry.metadata.is_multi_group:
        # Multi-Gruppe: kein SHM, keine Slot-Vorallokation
        # Nur den Pending-Write-Eintrag initialisieren für finish_write
        with self._pending_shm_lock:
            transfer_key = self._make_transfer_key(key, instance_id)
            if transfer_key not in self._pending_shm_writes:
                self._pending_shm_writes[transfer_key] = []
        return PrepareStoreResponse(slots=[])   # leere Slots = Pickle-Modus

    # Bestehender Pfad für Einzel-Gruppe
    strategy = self._strategies[instance_id]
    return strategy.prepare_store(key=key, instance_id=instance_id, ...)
```

---

## Zusammenfassung der Änderungen

| Datei | Art der Änderung |
|---|---|
| `custom_types.py` | `GroupLayoutInfo` Struct neu; `RegisterEngineDrivenContextPayload.group_layouts` Feld |
| `transfer_context/base.py` | `EngineDrivenContextMetadata` um per-Gruppe-Felder; `slice_kv_caches_for_group`; `gather_paged_kv_multi_group_to_cpu`; `scatter_cpu_multi_group_to_paged_kv`; `_serialize_multi_group_chunks`; `_deserialize_multi_group_chunks` |
| `transfer_context/worker_transfer.py` | `EngineDrivenTransferContext.register/submit_store/submit_retrieve` um Multi-Gruppen-Logik; `_single_group_block_ids` bleibt für Einzel-Gruppe |
| `modules/engine_driven_transfer.py` | `register_kv_cache_engine_driven_context`, `commit_store`, `prepare_retrieve` um Multi-Gruppen-Pfade |
| `modules/server_transfer.py` | Keine inhaltliche Änderung (optional: Guard-Assertion) |

**Nicht zu ändern:**
- `lmcache_mp_connector.py` (vLLM-Integration — unverändert)
- `platform/cuda/cache_context.py` (lmcache-driven Pfad — unverändert)
- `protocols/base.py` / `protocols/engine.py` (nur `PrepareRetrieveResponse` ggf. anpassen)
- Alle lmcache-driven Pfade (`lmcache_driven_transfer.py`, `gpu_connector/`, etc.)
