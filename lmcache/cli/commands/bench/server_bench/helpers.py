# SPDX-License-Identifier: Apache-2.0
"""Internal helpers for ``lmcache bench server``.

This module owns the heavy runtime imports (``torch`` / ``zmq`` /
``lmcache.v1.*``) and all pure / low-level helper functions used by
the ``server`` bench target. The CLI registration and execute
orchestration live in :mod:`lmcache.cli.commands.bench.server_bench.command`.

Splitting the module this way keeps the public command surface in line
with the ``engine_bench`` and ``l2_adapter_bench`` siblings, while
still quarantining the heavy imports behind a single guarded block so
the slim ``lmcache-cli`` install can load the bench parser without
torch / zmq.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Any
import ctypes
import hashlib
import json
import mmap
import sys
import time
import urllib.error
import urllib.request

# First Party
from lmcache import torch_dev, torch_device_type

# ``lmcache bench server`` allocates real CUDA tensors and talks to
# the MP server via ZMQ, both of which are absent from the thin
# ``lmcache-cli`` distribution (no torch, no zmq, no lmcache.v1.*).
# Importing them unconditionally would kill the *entire* ``lmcache``
# CLI at registry load time with an opaque ImportError. Wrap the
# heavy imports and remember the error so ``add_arguments`` /
# ``execute`` can bail out with an actionable install hint.
_IMPORT_ERROR: ImportError | None = None
try:
    # Third Party
    import torch
    import zmq  # noqa: F401  # availability probe; used by command.py

    # First Party
    from lmcache.utils import (
        EngineType,
        check_interprocess_event_support,
    )
    from lmcache.v1.kv_layer_groups import (
        DTYPE_MAP,
        KVLayerGroupInfo,
    )
    from lmcache.v1.multiprocess.custom_types import (
        IPCCacheServerKey,
        KVCache,
        RegisterEngineDrivenContextPayload,
    )
    from lmcache.v1.multiprocess.futures import MessagingFuture
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo
    from lmcache.v1.multiprocess.mq import MessageQueueClient
    from lmcache.v1.multiprocess.posix_shm import shm_open_pool_as_mmap
    from lmcache.v1.multiprocess.protocols.base import RequestType
    from lmcache.v1.multiprocess.protocols.engine import (
        RegisterEngineDrivenContextResponse,
    )
    from lmcache.v1.multiprocess.transfer_context.shm import ShmSlotDescriptor
    from lmcache.v1.platform.cpu.shm import (
        CpuShmTensorWrapper,
        shm_create_readwrite,
    )
except ImportError as _exc:
    _IMPORT_ERROR = _exc
    # Fallback placeholder so ``add_arguments`` can still build its
    # help text without crashing on a CLI-only install.
    DTYPE_MAP = {}  # type: ignore[assignment]

    # Stubs so other modules (notably ``command.py``) can still import
    # the SHM helpers on a slim install; ``_require_full_install`` is
    # the gate that prevents them from ever being invoked there.
    def shm_open_pool_as_mmap(name: str, nbytes: int) -> Any:  # type: ignore[misc]
        raise RuntimeError(
            "shm_open_pool_as_mmap unavailable on slim lmcache-cli install"
        )


def _require_full_install() -> None:
    """Exit with an install hint if the full LMCache runtime is missing.

    ``lmcache bench server`` needs torch, zmq and ``lmcache.v1.*``
    (MP client, KV layer-group parser). When those imports failed at
    module load — almost always because the user installed
    ``lmcache-cli`` instead of the full package — print the shortest
    actionable message to stderr and exit with status ``2`` so
    scripts can detect the install gap programmatically.
    """
    if _IMPORT_ERROR is None:
        return
    print(
        "ERROR: `lmcache bench server` needs the full LMCache package "
        "(torch, zmq, MP runtime), but only the `lmcache-cli` shell "
        "appears to be installed.\n"
        "  Install the full package with `pip install lmcache` and try "
        "again.\n"
        f"  Original import error: {_IMPORT_ERROR}",
        file=sys.stderr,
    )
    sys.exit(2)


# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

_HELLO_TOKEN_ID = 9906
_MODEL_NAME = "test-model"
_WORLD_SIZE = 1
_INSTANCE_ID = 0

# TP > 1 support: each simulated worker registers under a distinct
# ``instance_id`` so the server can hold one context per rank. Kept
# well above legacy ``_INSTANCE_ID = 0`` so single-worker bench runs
# and the multi-worker path never collide on the server side.
_INSTANCE_ID_BASE = 1000

# Default KV shape spec matching the original defaults:
# 32 layers, (2, num_blocks=1024, block_size=16, 8 heads, 128 head_size)
_DEFAULT_SHAPE_SPEC = "(2,1024,16,8,128):float16:32"


@dataclass
class WorkerContext:
    """Per-rank state for a single simulated TP worker.

    Mirrors what ``LMCacheMPWorkerAdapter`` holds per vLLM worker in a
    real deployment. Two identifiers matter here and they are *not* the
    same as the vLLM rank:

    * ``kv_worker_id`` is the identifier used in ``IPCCacheServerKey``.
      In vLLM this is ``ParallelStrategy.kv_worker_id`` --
      ``vllm_worker_id // tp_size`` under MLA (all TP ranks fold into
      one kv_worker) and ``vllm_worker_id`` otherwise.
    * ``kv_world_size`` is the world_size the server uses to index
      cache entries. Also from ``ParallelStrategy``: for MLA it is
      ``vllm_world_size // tp_size`` (typically 1 for pure TP-MLA)
      and ``vllm_world_size`` otherwise.

    ``instance_id`` stays per-simulated-rank so each vLLM process still
    gets its own ``REGISTER_KV_CACHE`` context; multiple TP ranks
    inside the same MLA kv_worker share the same ``kv_worker_id`` but
    keep distinct ``instance_id`` values.

    Attributes:
        kv_worker_id: The KV worker id (``ParallelStrategy.kv_worker_id``).
        kv_world_size: The KV world size (``ParallelStrategy.kv_world_size``).
        instance_id: Unique server-side context id for this simulated
            vLLM rank; matches what would be ``os.getpid()`` in a real
            deployment.
        client_tensors: Paged KV tensors owned by this worker (data-mode
            self-check source/sink). ``None`` in handle mode.
        server_pool: mmap of the server's SHM pool for this worker's
            engine-driven context. ``None`` outside data mode.
        is_kv_writer: Whether this worker participates in STORE. For MLA
            only ranks with ``vllm_worker_id % tp_size == 0`` write; non-
            MLA writes on every rank. RETRIEVE runs on every rank.
    """

    kv_worker_id: int
    kv_world_size: int
    instance_id: int
    client_tensors: "list[torch.Tensor] | None" = None
    server_pool: "mmap.mmap | None" = None
    is_kv_writer: bool = True


# ------------------------------------------------------------------ #
#  Low-level helpers                                                   #
# ------------------------------------------------------------------ #

# Default RPC call timeout (seconds) for blocking request/reply
# round-trips.
_DEFAULT_RPC_TIMEOUT_S = 10.0

# Unique sentinel returned by :func:`_call` on RPC timeout so callers
# can disambiguate it from a legitimate ``None`` (void) reply.
_TIMEOUT = object()


def _call(
    client: MessageQueueClient,
    request_type: RequestType,
    payloads: list,
    timeout_s: float = _DEFAULT_RPC_TIMEOUT_S,
) -> Any:
    """Submit a request through ``MessageQueueClient`` and block.

    Returns the decoded response (possibly ``None`` for void replies)
    on success, or the sentinel ``_TIMEOUT`` on RPC timeout.
    """
    future: MessagingFuture[Any] = client.submit_request(request_type, payloads)
    try:
        return future.result(timeout=timeout_s)
    except TimeoutError:
        return _TIMEOUT


# ------------------------------------------------------------------ #
#  Token / key helpers                                                 #
# ------------------------------------------------------------------ #


def _build_token_ids(
    seq_no: int,
    num_tokens: int,
) -> tuple[int, ...]:
    """Build token sequence: ``(seq_no, hello, hello, ...)``."""
    return (seq_no,) + (_HELLO_TOKEN_ID,) * num_tokens


def _make_key(
    token_ids: tuple[int, ...],
    request_id: str,
    start: int = 0,
    end: int = 0,
    worker_id: int | None = None,
    world_size: int = _WORLD_SIZE,
    num_kv_readers: int = 1,
) -> IPCCacheServerKey:
    """Build an IPCCacheServerKey."""
    return IPCCacheServerKey(
        model_name=_MODEL_NAME,
        world_size=world_size,
        num_kv_readers=num_kv_readers,
        worker_id=worker_id,
        token_ids=token_ids,
        start=start,
        end=end if end > 0 else len(token_ids),
        request_id=request_id,
    )


# ------------------------------------------------------------------ #
#  Protocol operations                                                 #
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  GPU KV cache allocation                                             #
# ------------------------------------------------------------------ #


# The server's vLLM detector identifies MLA layers by tensor rank: each
# layer must be rank-3 ``(NB, BS, HS)`` (see ``VLLM_Detector.discover``
# in ``lmcache/v1/gpu_connector/kv_format/detectors/vllm.py``). Classical
# split-K/V is rank-5 ``(2, NB, BS, NH, HS)``. Sharing this shape recipe
# across all allocation / gather / scatter helpers keeps the bench in
# sync with the detector contract regardless of transfer mode.
def _is_mla_kv_size(kv_size: int) -> bool:
    """``kv_size == 1`` marks a single-plane KV group (MLA / fused-K/V).

    Single source of truth for "is this group MLA?". Derived helpers
    (:func:`_make_alloc_shape`, :func:`_tensor_is_mla`) express the same
    contract in shape-space so that alloc / gather / scatter / checksum
    paths never diverge.
    """
    return kv_size == 1


def _tensor_is_mla(t: "torch.Tensor") -> bool:
    """Inverse of :func:`_is_mla_kv_size` at the tensor level.

    Client tensors produced by :func:`_make_alloc_shape` are rank-3
    ``(NB, BS, hidden)`` for MLA groups and rank-5
    ``(kv, NB, BS, NH, HS)`` for classical K/V groups. Checking
    ``dim() == 3`` here (instead of scattering the literal across
    gather / scatter / checksum) keeps every consumer routed through
    the same rule the allocator used.
    """
    return t.dim() == 3


def _make_alloc_shape(
    kv_size: int,
    num_blocks: int,
    block_size: int,
    num_heads: int,
    head_size: int,
) -> tuple[int, ...]:
    """Per-layer paged tensor shape, honouring the MLA rank-3 contract."""
    if _is_mla_kv_size(kv_size):
        return (num_blocks, block_size, num_heads * head_size)
    return (kv_size, num_blocks, block_size, num_heads, head_size)


def _group_alloc_shape(shape_desc) -> tuple[int, ...]:
    """``_make_alloc_shape`` variant reading fields off a ``shape_desc``."""
    return _make_alloc_shape(
        shape_desc.kv_size,
        shape_desc.nb,
        shape_desc.bs,
        shape_desc.nh,
        shape_desc.hs,
    )


def _allocate_gpu_kv_cache(
    num_layers: int = 32,
    num_heads: int = 8,
    head_size: int = 128,
    num_blocks: int = 1024,
    block_size: int = 16,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
    kv_size: int = 2,
    groups: list[KVLayerGroupInfo] | None = None,
) -> list[torch.Tensor]:
    """Allocate paged GPU KV cache tensors.

    Each layer is a tensor of shape
    ``(kv_size, num_blocks, block_size, num_heads, head_size)``
    matching the vLLM NHD layout. ``kv_size`` is 2 for standard
    K/V attention; override via the ``--kvcache-shape-spec``
    first dimension for architectures that need a different
    leading dimension (e.g. MLA).

    When ``groups`` is provided, tensors are allocated per-group
    using each group's own ``(kv_size, NB, BS, NH, HS)`` / ``dtype``
    (for heterogeneous multi-group specs). In that mode the flat
    ``num_heads`` / ``head_size`` / ``dtype`` / ``kv_size`` kwargs
    are ignored, and ``num_layers`` is derived from the groups.
    """
    # ``torch.float16`` cannot be used as a default value because the
    # module must load on ``lmcache-cli`` (no torch) installs.
    if dtype is None:
        dtype = torch.float16
    torch.random.manual_seed(42)
    dev = (
        torch.device(device)
        if device
        else torch.device(torch_device_type, torch_dev.current_device())
    )

    def _alloc(
        shape: tuple[int, ...],
        a_dtype: torch.dtype,
    ) -> torch.Tensor:
        if a_dtype.is_floating_point:
            return torch.randn(shape, dtype=a_dtype, device=dev)
        # ``torch.randn`` only supports floating-point dtypes; fall
        # back to ``randint`` for integer dtypes (e.g. ``uint8``
        # used by FP8 quantized KV cache layouts).
        iinfo = torch.iinfo(a_dtype)
        return torch.randint(iinfo.min, iinfo.max + 1, shape, dtype=a_dtype, device=dev)

    if groups:
        tensors: list[torch.Tensor] = []
        for g in groups:
            sd = g.shape_desc
            g_shape = _group_alloc_shape(sd)
            tensors.extend(_alloc(g_shape, g.dtype) for _ in range(sd.nl))
        return tensors

    shape = _make_alloc_shape(kv_size, num_blocks, block_size, num_heads, head_size)
    return [_alloc(shape, dtype) for _ in range(num_layers)]


# Backward-compatible alias used by tests and older callers.
_allocate_kv_cache = _allocate_gpu_kv_cache


def _allocate_cpu_shm_kv_cache(
    groups: list[KVLayerGroupInfo],
    shm_prefix: str,
) -> tuple[list[torch.Tensor], list[CpuShmTensorWrapper], list[str]]:
    """Allocate paged CPU KV cache tensors backed by POSIX SHM.

    For each (group, layer) we ``shm_open`` a fresh segment and
    ``mmap`` it into the client process. The returned tensors share
    storage with the SHM mapping, and the matching
    :class:`CpuShmTensorWrapper` instances tell the LMCache mp
    server how to map the very same physical pages -- i.e. true
    zero-copy across processes (matching the GPU CUDA-IPC path).

    Returns:
        ``(tensors, wrappers, shm_names)``. ``shm_names`` is kept
        so the caller can ``shm_unlink`` on shutdown.
    """
    # Fixed seed so the deterministic random fill below produces
    # reproducible checksums across cold/warm bench iterations.
    torch.random.manual_seed(42)
    tensors: list[torch.Tensor] = []
    wrappers: list[CpuShmTensorWrapper] = []
    shm_names: list[str] = []
    layer_idx = 0
    for g_idx, g in enumerate(groups):
        sd = g.shape_desc
        g_shape = _group_alloc_shape(sd)
        for _ in range(sd.nl):
            n_elems = 1
            for d in g_shape:
                n_elems *= d
            nbytes = n_elems * g.dtype.itemsize
            name = "%s_%d_%d" % (shm_prefix, g_idx, layer_idx)
            addr = shm_create_readwrite(name, nbytes)
            buf_type = ctypes.c_uint8 * nbytes
            buf = buf_type.from_address(addr)
            flat = torch.frombuffer(buf, dtype=torch.uint8)
            t = flat.view(g.dtype).reshape(g_shape)
            # Initialise with deterministic random data so the
            # cold/warm checksum compare in the bench loop is
            # meaningful.
            if g.dtype.is_floating_point:
                t.copy_(torch.randn(g_shape, dtype=g.dtype))
            else:
                iinfo = torch.iinfo(g.dtype)
                t.copy_(torch.randint(iinfo.min, iinfo.max + 1, g_shape, dtype=g.dtype))
            tensors.append(t)
            wrappers.append(CpuShmTensorWrapper(t, name))
            shm_names.append(name)
            layer_idx += 1
    return tensors, wrappers, shm_names


def _send_register_kv_cache(
    client: MessageQueueClient,
    instance_id: int = 0,
    model_name: str = _MODEL_NAME,
    world_size: int = _WORLD_SIZE,
    layout_hints: dict | None = None,
    kv_caches: KVCache | None = None,
    use_gpu: bool = True,
    use_handle: bool | None = None,
    engine_group_infos: "list[EngineGroupInfo] | None" = None,
) -> "bool | RegisterEngineDrivenContextResponse":
    """Register a KV cache context with the MP server.

    Dispatches to the correct protocol based on ``use_handle``:

    * Handle mode: ``REGISTER_KV_CACHE`` with a wrapper list
      (``CudaIPCWrapper`` for GPU, ``CpuShmTensorWrapper`` for CPU).
    * Data mode: ``REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`` with a
      ``RegisterEngineDrivenContextPayload`` derived from ``layout_hints``.

    ``use_handle`` defaults to ``use_gpu`` for backwards compatibility:
    GPU always goes through the handle path, CPU defaults to data.

    ``engine_group_infos`` (handle mode only) carries the per-group
    metadata — including each group's true ``tokens_per_block`` — so the
    server does not have to trust the block size discovered from the
    tensors (which the HND layout can swap with ``num_heads``). ``None``
    sends an empty list (single non-hybrid group, geometry discovered
    from the tensors).
    """
    if use_handle is None:
        use_handle = use_gpu
    if use_handle:
        if not kv_caches:
            raise ValueError(
                "kv_caches must be a non-empty list of wrappers "
                "(CudaIPCWrapper for GPU, CpuShmTensorWrapper for CPU)"
            )
        hints: dict = {"kv_layout": "NHD"}
        if layout_hints:
            hints.update(layout_hints)
        # TODO(maobaolong): Make the engine type configurable
        payloads = [
            instance_id,
            kv_caches,
            model_name,
            world_size,
            EngineType.VLLM,
            hints,
            list(engine_group_infos or ()),
        ]
        result = _call(client, RequestType.REGISTER_KV_CACHE, payloads)
        return result is not _TIMEOUT

    # CPU mode: use the non-GPU context registration protocol.
    # layout_hints carries num_layers, num_heads, head_size, block_size,
    # dtype, kv_size.  hidden_dim_size = num_heads * head_size (NHD).
    hints_d: dict = layout_hints or {}
    num_layers = int(hints_d.get("num_layers", 32))
    num_heads = hints_d.get("num_heads", 8)
    head_size = hints_d.get("head_size", 128)
    block_size = int(hints_d.get("block_size", 16))
    dtype_str = str(hints_d.get("dtype", "float16"))
    # "mixed" can appear for heterogeneous specs; fall back to first group.
    if not isinstance(num_heads, int):
        num_heads = 8
    if not isinstance(head_size, int):
        head_size = 128
    hidden_dim_size = int(num_heads) * int(head_size)
    # ``kv_size`` == 1 marks an MLA group (single-plane KV: no separate
    # K/V leading dim). The server uses ``use_mla`` to decide whether
    # the SHM chunk shape is ``(NL, chunk, hidden)`` or
    # ``(2, NL, chunk, hidden)``. ``"mixed"`` (heterogeneous specs) is
    # not representable in a single data-mode register, so we default
    # to non-MLA in that case.
    kv_size_hint = hints_d.get("kv_size", 2)
    use_mla = isinstance(kv_size_hint, int) and _is_mla_kv_size(kv_size_hint)
    payload = RegisterEngineDrivenContextPayload(
        instance_id=instance_id,
        model_name=model_name,
        world_size=world_size,
        block_size=block_size,
        num_layers=num_layers,
        hidden_dim_size=hidden_dim_size,
        dtype_str=dtype_str,
        use_mla=use_mla,
    )
    result = _call(
        client, RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT, [payload]
    )
    if result is _TIMEOUT:
        return False
    # The data-mode register reply carries the server's SHM pool name
    # and size; the bench keeps it on the side so STORE / RETRIEVE
    # can mmap the same pool and exchange tensor data without going
    # through pickle.
    return result


def _send_unregister_kv_cache(
    client: MessageQueueClient,
    instance_id: int = 0,
    use_handle: bool = True,
) -> bool:
    """Deregister a KV cache context from the MP server.

    The inverse of :func:`_send_register_kv_cache`. Without this call
    the server keeps the bench's registration (and the CUDA-IPC / POSIX
    SHM mappings it holds) alive forever, leaking one context entry per
    bench run.

    Dispatches to the correct protocol based on ``use_handle``, mirroring
    the register path:

    * Handle mode: ``UNREGISTER_KV_CACHE``.
    * Data mode: ``UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT``.

    Both protocols take a single ``instance_id`` payload and return a void
    reply, so success is distinguished from an RPC timeout only.

    Args:
        client: The MP message-queue client.
        instance_id: The instance ID used at registration time. Must match
            the ``instance_id`` passed to :func:`_send_register_kv_cache`.
        use_handle: ``True`` for the handle path (GPU CUDA-IPC / CPU SHM),
            ``False`` for the engine-driven data path.

    Returns:
        ``True`` if the server acknowledged the call, ``False`` on RPC
        timeout.
    """
    request_type = (
        RequestType.UNREGISTER_KV_CACHE
        if use_handle
        else RequestType.UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT
    )
    result = _call(client, request_type, [instance_id])
    return result is not _TIMEOUT


def _send_lookup(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
    tp_size: int = 1,
) -> bool:
    """LOOKUP — submit a prefix lookup.

    The server reserves ``key.num_kv_readers`` read locks per chunk
    (each reader's RETRIEVE releases one; see
    ``IPCCacheServerKey.require_num_kv_readers``). ``tp_size`` is
    a legacy wire field the server ignores.

    The server-side handler returns ``None`` (void) on success, so
    we only distinguish RPC timeout from a completed call.
    """
    result = _call(client, RequestType.LOOKUP, [key, tp_size])
    return result is not _TIMEOUT


def _poll_prefetch_status(
    client: MessageQueueClient,
    request_id: str,
    max_polls: int = 50,
    poll_interval: float = 0.05,
) -> int | None:
    """QUERY_PREFETCH_STATUS — poll until done.

    Returns the hit chunk count, or ``None`` if the polling budget
    is exhausted. The server keys prefetch jobs by ``request_id``
    (str), not an integer job handle.
    """
    for _ in range(max_polls):
        result = _call(
            client,
            RequestType.QUERY_PREFETCH_STATUS,
            [request_id],
        )
        if result is _TIMEOUT:
            # RPC timeout — treat as giving up on this poll cycle.
            return None
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


def _make_event_handle(use_gpu: bool = True) -> bytes:
    """Create a CUDA event IPC handle for GPU mode.

    CPU mode does not need a cross-process event (SHM mappings are
    coherent without device-side sync), so an empty handle is
    returned and the server treats it as a no-op.
    """
    if not use_gpu:
        return b""
    check_interprocess_event_support()
    event = torch_dev.Event(interprocess=True)
    event.record()
    return event.ipc_handle()


def _build_server_slot_views(
    server_pool: "mmap.mmap",
    slots: list[dict[str, Any]],
) -> list["torch.Tensor"]:
    """Build zero-copy tensor views over server SHM slot descriptors.

    Each ``ShmSlotDescriptor`` carries the ``(offset, length, shape,
    dtype)`` of one chunk inside the server-owned SHM pool; we wrap
    them with ``torch.frombuffer`` so the bench can read or overwrite
    that chunk without going through pickle.
    """
    views: list[torch.Tensor] = []
    for raw in slots:
        desc = ShmSlotDescriptor.from_dict(raw)
        dtype = getattr(torch, desc.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError("invalid torch dtype string: %s" % desc.dtype)
        itemsize = torch.empty((), dtype=dtype).element_size()
        if itemsize <= 0:
            raise ValueError("invalid dtype size for %s" % desc.dtype)
        count = desc.length // itemsize
        flat = torch.frombuffer(
            server_pool, dtype=dtype, count=count, offset=desc.offset
        )
        views.append(flat.view(torch.Size(desc.shape)))
    return views


def _gather_paged_to_flat_chunks(
    tensors: list["torch.Tensor"],
    block_offset: int,
    num_blocks: int,
    block_size: int,
    chunk_size: int,
) -> list["torch.Tensor"]:
    """Gather paged client tensors into flat per-chunk CPU tensors.

    Output layout matches the server's expected ``commit_store``
    payload (set up at register time by
    ``register_kv_cache_engine_driven_context``):
    each chunk is ``[2, num_layers, chunk_size, hidden_dim]``,
    where ``hidden_dim = NH * HS``. Assumes a homogeneous group
    (same NH/HS/dtype across all layers); heterogeneous specs
    fall outside the bench scope.
    """
    if chunk_size % block_size != 0:
        raise ValueError(
            "chunk_size %d must be a multiple of block_size %d"
            % (chunk_size, block_size)
        )
    blocks_per_chunk = chunk_size // block_size
    num_chunks = num_blocks // blocks_per_chunk
    num_layers = len(tensors)
    # Client tensors are rank-3 ``(NB, BS, hidden)`` in MLA mode and
    # rank-5 ``(kv, NB, BS, NH, HS)`` otherwise (see
    # :func:`_tensor_is_mla`). The block-axis lives at dim 0 for MLA
    # and dim 1 for classical; per-layer flats stack into a 3D or 4D
    # chunk to match the server's single-plane / split-K/V commit shape.
    first_is_mla = bool(tensors) and _tensor_is_mla(tensors[0])
    chunks: list[torch.Tensor] = []
    for c in range(num_chunks):
        start_b = block_offset + c * blocks_per_chunk
        per_layer: list[torch.Tensor] = []
        for t in tensors:
            if _tensor_is_mla(t):
                # MLA: (NB, BS, hidden) -> (chunk_size, hidden).
                sliced = t.narrow(0, start_b, blocks_per_chunk)
                _, bs, hidden = sliced.shape
                flat = sliced.contiguous().view(blocks_per_chunk * bs, hidden)
            else:
                # Classical: (kv, NB, BS, NH, HS) -> (kv, chunk_size, NH*HS).
                sliced = t.narrow(1, start_b, blocks_per_chunk)
                kv, _, bs, nh, hs = sliced.shape
                flat = sliced.contiguous().view(kv, blocks_per_chunk * bs, nh * hs)
            per_layer.append(flat)
        # MLA per-layer flats are 2D; classical are 3D. Stack picks the
        # right rank automatically: dim=0 for MLA yields (NL, chunk, hidden);
        # dim=1 for classical yields (kv, NL, chunk, hidden).
        stack_dim = 0 if first_is_mla else 1
        chunk = torch.stack(per_layer, dim=stack_dim).contiguous()
        if chunk.shape[stack_dim] != num_layers:
            raise RuntimeError(
                "unexpected chunk shape %s (NL mismatch)" % (chunk.shape,)
            )
        chunks.append(chunk)
    return chunks


def _scatter_flat_chunks_to_paged(
    tensors: list["torch.Tensor"],
    chunks: list["torch.Tensor"],
    block_offset: int,
    block_size: int,
    chunk_size: int,
) -> None:
    """Inverse of :func:`_gather_paged_to_flat_chunks`.

    Writes each ``[2, NL, chunk_size, hidden]`` flat chunk back into
    the paged client tensors at the matching block range. Used by
    the data-mode RETRIEVE path so the bench's client-side checksum
    can compare cold ground truth with what the server returned.
    """
    if chunk_size % block_size != 0:
        raise ValueError(
            "chunk_size %d must be a multiple of block_size %d"
            % (chunk_size, block_size)
        )
    blocks_per_chunk = chunk_size // block_size
    for c, chunk in enumerate(chunks):
        start_b = block_offset + c * blocks_per_chunk
        # MLA chunks are 3D ``(NL, chunk, hidden)`` (kv_size == 1 is
        # folded away by the server), classical K/V chunks are 4D
        # ``(kv, NL, chunk, hidden)``. Client tensors match: MLA rank-3
        # ``(NB, BS, hidden)`` vs. classical rank-5 ``(kv, NB, BS, NH, HS)``
        # -- both derived from the same :func:`_is_mla_kv_size` contract
        # via :func:`_tensor_is_mla`.
        chunk_is_mla = _tensor_is_mla(chunk)
        for layer_idx, t in enumerate(tensors):
            if _tensor_is_mla(t):
                # MLA: block axis at dim 0.
                target = t.narrow(0, start_b, blocks_per_chunk)
                flat = chunk[layer_idx] if chunk_is_mla else chunk[:, layer_idx]
                nb, bs, hidden = target.shape
                target.copy_(flat.reshape(nb, bs, hidden))
            else:
                kv, _, bs, nh, hs = t.shape
                target = t.narrow(1, start_b, blocks_per_chunk)
                flat = chunk[layer_idx] if chunk_is_mla else chunk[:, layer_idx]
                target.copy_(flat.reshape(kv, blocks_per_chunk, bs, nh, hs))


# ------------------------------------------------------------------ #
#  Client-side checksum / zero-fill (data-mode self-check)             #
# ------------------------------------------------------------------ #


def _compute_client_checksums(
    tensors: list["torch.Tensor"],
    block_offset: int,
    num_blocks: int,
    block_size: int,
    chunk_size: int,
) -> list[str]:
    """Hash a paged block range from client-side KV tensors.

    For each chunk (``chunk_size // block_size`` consecutive blocks),
    feed every layer's bytes for that block range into a single MD5
    digest. The returned list maps 1:1 to the chunks the bench loop
    expects, so a cold-pass digest can be compared with a warm-pass
    digest to verify that ``RETRIEVE`` actually wrote back the data
    we wrote during ``STORE`` -- without relying on a server-side
    ``/cache/checksums`` endpoint (which only exists in handle mode).
    """
    if chunk_size % block_size != 0:
        raise ValueError(
            "chunk_size %d must be a multiple of block_size %d"
            % (chunk_size, block_size)
        )
    blocks_per_chunk = chunk_size // block_size
    num_chunks = num_blocks // blocks_per_chunk
    checksums: list[str] = []
    for c in range(num_chunks):
        start_b = block_offset + c * blocks_per_chunk
        end_b = start_b + blocks_per_chunk
        h = hashlib.md5()
        for t in tensors:
            # Block axis is dim 0 for MLA rank-3 tensors ``(NB, BS, hidden)``
            # and dim 1 for classical rank-5 ``(kv, NB, BS, NH, HS)``.
            # ``contiguous().numpy().tobytes()`` survives non-contiguous
            # slices and dtype quirks (bfloat16 has no numpy view, but
            # uint8 reinterpret works after slice).
            block_dim = 0 if _tensor_is_mla(t) else 1
            view = t.narrow(block_dim, start_b, end_b - start_b).contiguous()
            h.update(view.view(torch.uint8).numpy().tobytes())
        checksums.append(h.hexdigest())
    return checksums


def _zero_fill_client_blocks(
    tensors: list["torch.Tensor"],
    block_offset: int,
    num_blocks: int,
) -> None:
    """Zero out a paged block range across all client tensors.

    Used right before a warm-pass ``RETRIEVE`` so that any non-zero
    bytes observed afterwards must have been written by the server.
    Without this, a warm checksum equal to the cold checksum could
    still happen even if ``RETRIEVE`` was a silent no-op (the SHM
    pages were never overwritten in the first place).
    """
    for t in tensors:
        block_dim = 0 if _tensor_is_mla(t) else 1
        t.narrow(block_dim, block_offset, num_blocks).zero_()


def _send_store(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
    block_offset: int = 0,
    block_size: int = 16,
    num_engine_group_infos: int = 1,
    use_gpu: bool = True,
    use_handle: bool | None = None,
    client_tensors: list["torch.Tensor"] | None = None,
    chunk_size: int = 0,
    server_pool: "mmap.mmap | None" = None,
    instance_id: int = _INSTANCE_ID,
) -> str:
    """Store KV cache blocks. Returns status string.

    Handle mode uses the single-shot ``STORE`` RPC (GPU CUDA-IPC, or
    CPU SHM with an empty event handle).
    Data mode uses the two-phase ``PREPARE_STORE`` + ``COMMIT_STORE``.
    When ``server_pool`` and ``client_tensors`` are both supplied the
    bench gathers the paged block range into flat per-chunk CPU
    tensors and writes them straight into the server-owned SHM pool
    via the slot descriptors returned by ``PREPARE_STORE``, so the
    follow-up ``COMMIT_STORE`` carries an empty payload and the
    server stays on its zero-copy SHM path.
    """
    if use_handle is None:
        use_handle = use_gpu
    if use_handle:
        num_tokens = key.end - key.start
        num_blocks = num_tokens // block_size
        block_ids = list(range(block_offset, block_offset + num_blocks))
        payloads = [
            key,
            instance_id,
            [block_ids] * num_engine_group_infos,
            _make_event_handle(),
        ]
        result = _call(client, RequestType.STORE, payloads)
        if result is _TIMEOUT:
            return "timeout"
        return "stored" if result[1] else "store_failed"

    # CPU mode: PREPARE_STORE -> COMMIT_STORE
    prep = _call(client, RequestType.PREPARE_STORE, [key, instance_id])
    if prep is _TIMEOUT:
        return "timeout"
    if server_pool is not None and client_tensors is not None and chunk_size > 0:
        ctx = prep.context if isinstance(prep.context, dict) else {}
        slots = ctx.get("slots", []) or []
        chunk_indices = ctx.get("chunk_indices", []) or []
        if slots and chunk_indices:
            num_blocks = (key.end - key.start) // block_size
            full_chunks = _gather_paged_to_flat_chunks(
                client_tensors,
                block_offset,
                num_blocks,
                block_size,
                chunk_size,
            )
            slot_views = _build_server_slot_views(server_pool, slots)
            for slot_view, chunk_idx in zip(slot_views, chunk_indices, strict=False):
                if 0 <= chunk_idx < len(full_chunks):
                    slot_view.copy_(full_chunks[chunk_idx].view(slot_view.shape))
    commit = _call(client, RequestType.COMMIT_STORE, [key, instance_id, b""])
    if commit is _TIMEOUT:
        return "timeout"
    return "stored" if commit else "store_failed"


def _send_retrieve(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
    chunk_size: int,
    hit_chunks: int,
    block_offset: int = 0,
    block_size: int = 16,
    num_engine_group_infos: int = 1,
    use_gpu: bool = True,
    use_handle: bool | None = None,
    client_tensors: list["torch.Tensor"] | None = None,
    server_pool: "mmap.mmap | None" = None,
    instance_id: int = _INSTANCE_ID,
) -> str:
    """Retrieve KV cache blocks. Returns status.

    Handle mode uses the single-shot ``RETRIEVE`` RPC (GPU CUDA-IPC, or
    CPU SHM with an empty event handle).
    Data mode uses the two-phase ``PREPARE_RETRIEVE`` +
    ``COMMIT_RETRIEVE``. When ``server_pool`` and ``client_tensors``
    are both supplied the bench builds zero-copy tensor views over
    the slot descriptors returned by ``PREPARE_RETRIEVE`` and
    scatters them back into the paged client SHM, so the round-trip
    self-check can run without ``PREPARE_RETRIEVE`` having to ship a
    pickled copy of the chunks.
    """
    if use_handle is None:
        use_handle = use_gpu
    if use_handle:
        hit_tokens = hit_chunks * chunk_size
        num_blocks = hit_tokens // block_size
        block_ids = list(range(block_offset, block_offset + num_blocks))
        payloads = [
            key,
            instance_id,
            [block_ids] * num_engine_group_infos,
            _make_event_handle(),
            0,  # skip_first_n_tokens
        ]
        result = _call(client, RequestType.RETRIEVE, payloads)
        if result is _TIMEOUT:
            return "timeout"
        return "retrieved" if result[1] else "retrieve_failed"

    # CPU mode: PREPARE_RETRIEVE -> COMMIT_RETRIEVE
    prep = _call(client, RequestType.PREPARE_RETRIEVE, [key, instance_id])
    if prep is _TIMEOUT:
        return "timeout"
    if not prep.success:
        return "retrieve_failed"
    if server_pool is not None and client_tensors is not None:
        ctx = prep.context if isinstance(prep.context, dict) else {}
        slots = ctx.get("slots", []) or []
        if slots:
            try:
                slot_views = _build_server_slot_views(server_pool, slots)
                _scatter_flat_chunks_to_paged(
                    client_tensors,
                    slot_views,
                    block_offset,
                    block_size,
                    chunk_size,
                )
            except (RuntimeError, ValueError) as exc:
                print("  [WARNING] retrieve scatter failed: %s" % exc)
    commit = _call(client, RequestType.COMMIT_RETRIEVE, [key, instance_id])
    if commit is _TIMEOUT:
        return "timeout"
    return "retrieved" if commit else "retrieve_failed"


def _send_end_session(
    client: MessageQueueClient,
    request_id: str,
) -> None:
    """END_SESSION — clean up server-side session state."""
    _call(client, RequestType.END_SESSION, [request_id])


# ------------------------------------------------------------------ #
#  Checksum query                                                      #
# ------------------------------------------------------------------ #


def _query_checksum(
    http_base: str,
    block_offset: int,
    num_blocks: int,
    block_size: int,
    chunk_size: int,
    instance_id: int = _INSTANCE_ID,
) -> list[str] | None:
    """Query KV cache checksums via the HTTP API.

    Uses the MP-native ``block_ids`` + ``block_size`` addressing
    scheme so the query matches the same block-level semantics
    as ``STORE`` / ``RETRIEVE``. This CLI pins ``layerwise=false``
    so the server always returns ``chunk_checksums`` as a flat
    ``list[str]``. We still defensively validate the response
    type — if a future endpoint variant returns a per-layer
    ``dict`` we log and skip the comparison rather than letting
    ``str.join`` crash.

    ``instance_id`` selects the GPU context to hash. TP > 1 registers
    one context per rank so the caller must pass a real, registered
    ``instance_id`` — the default of ``0`` only works for the single
    legacy worker path.
    """
    blocks = list(range(block_offset, block_offset + num_blocks))
    # The MP /cache/checksums endpoint is block-native: its chunk_size counts
    # blocks per chunk, while our caller passes in the server-side token-level
    # chunk_size. Convert here.
    if chunk_size % block_size != 0:
        print(
            "  [WARNING] chunk_size %d not a multiple of block_size %d; "
            "skipping checksum query" % (chunk_size, block_size)
        )
        return None
    chunk_size_blocks = chunk_size // block_size
    url = "%s/cache/checksums" % http_base
    payload = json.dumps(
        {
            "block_ids": blocks,
            "chunk_size": chunk_size_blocks,
            "instance_id": instance_id,
            "layerwise": False,
        }
    ).encode()
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            if data.get("status") != "success":
                return None
            checksums = data.get("chunk_checksums", [])
            if not isinstance(checksums, list) or not all(
                isinstance(c, str) for c in checksums
            ):
                print(
                    "  [WARNING] unexpected chunk_checksums "
                    "type=%s; expected list[str]" % type(checksums).__name__
                )
                return None
            return checksums
    except (urllib.error.URLError, OSError) as exc:
        print("  [WARNING] Checksum query failed: %s" % exc)
    return None


# ------------------------------------------------------------------ #
#  Per-request flow                                                    #
# ------------------------------------------------------------------ #


@dataclass
class RequestResult:
    """Result of a single request pass (cold or warm).

    Carries both the checksum list (for correctness verification) and
    per-operation latency measurements (for metrics aggregation).
    """

    checksums: list[str] | None = None
    lookup_ms: float | None = None
    retrieve_ms: float | None = None
    store_ms: float | None = None
    hit_chunks: int = 0
    total_chunks: int = 0
    store_tokens: int = 0
    retrieve_tokens: int = 0


def _process_request(
    client: MessageQueueClient,
    seq_no: int,
    num_tokens: int,
    chunk_size: int,
    pass_label: str,
    http_base: str = "",
    block_size: int = 16,
    total_blocks: int = 1024,
    num_engine_group_infos: int = 1,
    use_gpu: bool = True,
    use_handle: bool | None = None,
    client_tensors: list["torch.Tensor"] | None = None,
    server_pool: "mmap.mmap | None" = None,
    workers: "list[WorkerContext] | None" = None,
    world_size: int = _WORLD_SIZE,
) -> RequestResult | None:
    """Run the full lookup -> retrieve/store flow.

    When ``client_tensors`` is provided (data-mode self-check), the
    flow gains two extra steps:

    * cold pass: hash the paged block range *before* ``STORE``, so
      the digest captures the ground-truth KV bytes.
    * warm pass: zero-fill the same block range *before*
      ``RETRIEVE``, then hash *after* ``RETRIEVE``. cold == warm
      proves the server returned the exact bytes we sent.

    Handle mode keeps the historical server-side
    ``/cache/checksums`` path; client tensors are not consulted (in
    handle mode the client and server share the same SHM/IPC
    pages, so a client-side hash equals itself by construction).

    ``workers`` (optional) drives multi-rank TP simulation: LOOKUP
    stays scheduler-scoped (single call, worker_id=None) while
    STORE fans out to every ``is_kv_writer`` rank and RETRIEVE fans
    out to every rank -- mirroring how ``LMCacheMPWorkerAdapter``
    routes requests in a real vLLM deployment. When ``None`` the
    bench synthesises a single worker from ``client_tensors`` /
    ``server_pool`` so single-rank runs stay unchanged.
    """
    # Materialise a single-worker context when the caller passed the
    # legacy client_tensors / server_pool pair. Keeps the fan-out loop
    # below oblivious to how many workers there are.
    if workers is None:
        workers = [
            WorkerContext(
                kv_worker_id=0,
                kv_world_size=world_size,
                instance_id=_INSTANCE_ID,
                client_tensors=client_tensors,
                server_pool=server_pool,
                is_kv_writer=True,
            )
        ]
    # ``client_tensors is not None`` gates client-side self-check (data
    # mode); use any worker's tensors as the flag since either all or
    # none of them carry tensors.
    any_client_tensors = workers[0].client_tensors is not None

    token_ids = _build_token_ids(seq_no, num_tokens)
    request_id = "req-%d-%s" % (seq_no, pass_label)

    # Align end to chunk_size (only full chunks)
    num_full_tokens = (len(token_ids) // chunk_size) * chunk_size
    if num_full_tokens == 0:
        print(
            "  [seq %d/%s] SKIP: %d tokens < chunk_size %d"
            % (seq_no, pass_label, len(token_ids), chunk_size)
        )
        return None

    # Key for lookup (worker_id=None)
    lookup_key = _make_key(
        token_ids,
        request_id,
        start=0,
        end=num_full_tokens,
        world_size=world_size,
    )

    # 1. LOOKUP
    # ``tp_size = len(workers)`` so MLA runs (kv_world_size==1,
    # tp_size>1) get ``tp_size-1`` extra read locks per chunk on the
    # server; without those every subsequent-rank RETRIEVE would
    # release a lock the first rank already dropped.
    t0 = time.monotonic()
    if not _send_lookup(client, lookup_key, tp_size=len(workers)):
        print("  [seq %d/%s] LOOKUP timeout" % (seq_no, pass_label))
        return None

    # 2. QUERY_PREFETCH_STATUS (poll by request_id)
    hit_chunks = _poll_prefetch_status(client, lookup_key.request_id)
    if hit_chunks is None:
        hit_chunks = 0

    total_chunks = num_full_tokens // chunk_size
    miss_chunks = total_chunks - hit_chunks
    hit_tokens = hit_chunks * chunk_size
    lookup_ms = (time.monotonic() - t0) * 1000

    print(
        "  [seq %d/%s] LOOKUP: %d/%d chunks hit "
        "(%.1f ms)"
        % (
            seq_no,
            pass_label,
            hit_chunks,
            total_chunks,
            lookup_ms,
        )
    )

    # Block offset: each request uses a different block
    # range so that different requests touch different data.
    # Wrap with modulo and clamp so the entire range
    # [block_offset, block_offset + num_blocks) stays
    # within [0, total_blocks).
    num_blocks = num_full_tokens // block_size
    usable = max(total_blocks - num_blocks, 1)
    block_offset = (seq_no * num_blocks) % usable

    # Client-side self-check (data mode only): cold pass captures
    # per-rank ground truth before STORE; warm pass zero-fills so
    # a successful RETRIEVE must overwrite every byte on each rank.
    cold_ground_truth: list[str] | None = None
    if any_client_tensors:
        if pass_label == "cold" and miss_chunks > 0:
            store_block_off = block_offset + (hit_tokens // block_size)
            store_num_blocks = (num_full_tokens - hit_tokens) // block_size
            cold_parts: list[str] = []
            for w in workers:
                if not w.is_kv_writer or w.client_tensors is None:
                    continue
                cold_parts.extend(
                    _compute_client_checksums(
                        w.client_tensors,
                        store_block_off,
                        store_num_blocks,
                        block_size,
                        chunk_size,
                    )
                )
            cold_ground_truth = cold_parts or None
        if pass_label == "warm" and hit_chunks > 0:
            retr_num_blocks = hit_tokens // block_size
            for w in workers:
                if w.client_tensors is not None:
                    _zero_fill_client_blocks(
                        w.client_tensors,
                        block_offset,
                        retr_num_blocks,
                    )

    # 3. RETRIEVE hit portion — every rank retrieves its own KV shard.
    retrieve_ms: float = 0.0
    store_ms: float = 0.0
    if hit_chunks > 0:
        t1 = time.monotonic()
        status = "retrieved"
        for w in workers:
            retrieve_key = _make_key(
                token_ids,
                request_id,
                start=0,
                end=hit_tokens,
                worker_id=w.kv_worker_id,
                world_size=world_size,
            )
            w_status = _send_retrieve(
                client,
                retrieve_key,
                chunk_size,
                hit_chunks,
                block_offset=block_offset,
                block_size=block_size,
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                client_tensors=w.client_tensors,
                server_pool=w.server_pool,
                instance_id=w.instance_id,
            )
            if w_status != "retrieved":
                status = w_status
        retrieve_ms = (time.monotonic() - t1) * 1000
        print(
            "  [seq %d/%s] RETRIEVE: %s "
            "(%d tokens, %.1f ms, %d workers)"
            % (
                seq_no,
                pass_label,
                status,
                hit_tokens,
                retrieve_ms,
                len(workers),
            )
        )

    # 4. STORE miss portion — only KV writers (all ranks non-MLA,
    # rank 0 only under MLA) send STORE.
    if miss_chunks > 0:
        store_start = hit_tokens
        store_end = num_full_tokens
        t2 = time.monotonic()
        store_block_off = block_offset + (hit_tokens // block_size)
        status = "stored"
        n_store_workers = 0
        for w in workers:
            if not w.is_kv_writer:
                continue
            n_store_workers += 1
            store_key = _make_key(
                token_ids,
                request_id,
                start=store_start,
                end=store_end,
                worker_id=w.kv_worker_id,
                world_size=world_size,
            )
            w_status = _send_store(
                client,
                store_key,
                block_offset=store_block_off,
                block_size=block_size,
                num_engine_group_infos=num_engine_group_infos,
                use_gpu=use_gpu,
                use_handle=use_handle,
                client_tensors=w.client_tensors,
                chunk_size=chunk_size,
                server_pool=w.server_pool,
                instance_id=w.instance_id,
            )
            if w_status != "stored":
                status = w_status
        store_ms = (time.monotonic() - t2) * 1000
        print(
            "  [seq %d/%s] STORE: %s "
            "(%d tokens, %.1f ms, %d writers)"
            % (
                seq_no,
                pass_label,
                status,
                store_end - store_start,
                store_ms,
                n_store_workers,
            )
        )

    # 5. Compute checksums.
    #   * data mode (workers have client_tensors):
    #       cold -> ground truth captured pre-STORE (writer ranks only,
    #               matching what the server actually receives)
    #       warm -> hash post-RETRIEVE on writer ranks so cold == warm
    #               proves each writer's bytes made a lossless round trip.
    #   * handle mode: query /cache/checksums on the server, which
    #     reads the shared SHM/IPC pages directly.
    checksums: list[str] | None = None
    if any_client_tensors and num_full_tokens > 0:
        if pass_label == "cold":
            checksums = cold_ground_truth
        elif pass_label == "warm" and hit_chunks > 0:
            retr_num_blocks = hit_tokens // block_size
            warm_parts: list[str] = []
            for w in workers:
                if not w.is_kv_writer or w.client_tensors is None:
                    continue
                warm_parts.extend(
                    _compute_client_checksums(
                        w.client_tensors,
                        block_offset,
                        retr_num_blocks,
                        block_size,
                        chunk_size,
                    )
                )
            checksums = warm_parts or None
    elif http_base and num_full_tokens > 0:
        # Handle-mode server-side hash. All ranks share the same paged
        # bytes for a given block range (each rank stored the same
        # data), so any registered instance_id gives the same digest;
        # pick the first worker's for simplicity.
        checksums = _query_checksum(
            http_base,
            block_offset,
            num_blocks,
            block_size,
            chunk_size,
            instance_id=workers[0].instance_id,
        )
    if checksums:
        digest = hashlib.md5("".join(checksums).encode()).hexdigest()[:16]
        print(
            "  [seq %d/%s] CHECKSUM: %s (%d chunks)"
            % (
                seq_no,
                pass_label,
                digest,
                len(checksums),
            )
        )

    # 6. END_SESSION
    _send_end_session(client, request_id)
    return RequestResult(
        checksums=checksums,
        lookup_ms=lookup_ms,
        retrieve_ms=retrieve_ms if hit_chunks > 0 else None,
        store_ms=store_ms if miss_chunks > 0 else None,
        hit_chunks=hit_chunks,
        total_chunks=total_chunks,
        store_tokens=(num_full_tokens - hit_tokens) if miss_chunks > 0 else 0,
        retrieve_tokens=hit_tokens if hit_chunks > 0 else 0,
    )


# ------------------------------------------------------------------ #
#  Server query helper                                                 #
# ------------------------------------------------------------------ #


def _get_chunk_size(client: MessageQueueClient) -> int:
    """Query the server's chunk size."""
    result = _call(client, RequestType.GET_CHUNK_SIZE, [])
    if result is _TIMEOUT or result is None:
        return 256  # fallback
    return int(result)
