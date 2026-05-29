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
from typing import Any
import hashlib
import json
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
        compress_slot_mapping,
    )
    from lmcache.v1.kv_layer_groups import (
        DTYPE_MAP,
        KVLayerGroupInfo,
    )
    from lmcache.v1.multiprocess.custom_types import (
        CudaIPCWrapper,
        IPCCacheEngineKey,
    )
    from lmcache.v1.multiprocess.futures import MessagingFuture
    from lmcache.v1.multiprocess.mq import MessageQueueClient
    from lmcache.v1.multiprocess.protocols.base import RequestType
except ImportError as _exc:
    _IMPORT_ERROR = _exc
    # Fallback placeholder so ``add_arguments`` can still build its
    # help text without crashing on a CLI-only install.
    DTYPE_MAP = {}  # type: ignore[assignment]


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

# Default KV shape spec matching the original defaults:
# 32 layers, (2, num_blocks=1024, block_size=16, 8 heads, 128 head_size)
_DEFAULT_SHAPE_SPEC = "(2,1024,16,8,128):float16:32"

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
) -> IPCCacheEngineKey:
    """Build an IPCCacheEngineKey."""
    return IPCCacheEngineKey(
        model_name=_MODEL_NAME,
        world_size=_WORLD_SIZE,
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
            g_shape = (sd.kv_size, sd.nb, sd.bs, sd.nh, sd.hs)
            tensors.extend(_alloc(g_shape, g.dtype) for _ in range(sd.nl))
        return tensors

    shape = (kv_size, num_blocks, block_size, num_heads, head_size)
    return [_alloc(shape, dtype) for _ in range(num_layers)]


def _send_register_kv_cache(
    client: MessageQueueClient,
    instance_id: int = 0,
    model_name: str = _MODEL_NAME,
    world_size: int = _WORLD_SIZE,
    layout_hints: dict | None = None,
    gpu_tensors: list[torch.Tensor] | None = None,
) -> bool:
    """REGISTER_KV_CACHE — register a KV cache context.

    In GPU mode real CUDA tensors are wrapped via
    ``CudaIPCWrapper`` and sent over IPC.

    .. note::
        CPU mode (``gpu_tensors is None``) is not yet
        supported.
    """
    hints: dict = {"kv_layout": "NHD"}
    if layout_hints:
        hints.update(layout_hints)

    if gpu_tensors is None:
        # TODO(maobaolong): support CPU mode registration
        raise NotImplementedError(
            "CPU mode is not yet supported. Please use --mode gpu."
        )

    kv_caches = [CudaIPCWrapper(t) for t in gpu_tensors]
    # TODO(maobaolong): Make the engine type configurable
    payloads = [
        instance_id,
        kv_caches,
        model_name,
        world_size,
        EngineType.VLLM,
        hints,
    ]
    result = _call(client, RequestType.REGISTER_KV_CACHE, payloads)
    return result is not _TIMEOUT


def _send_lookup(
    client: MessageQueueClient,
    key: IPCCacheEngineKey,
) -> bool:
    """LOOKUP — submit a prefix lookup.

    The server-side handler returns ``None`` (void) on success, so
    we only distinguish RPC timeout from a completed call.
    """
    result = _call(client, RequestType.LOOKUP, [key, 1])
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


def _make_event_handle() -> bytes:
    """Create a CUDA event IPC handle for GPU mode."""
    check_interprocess_event_support()
    event = torch_dev.Event(interprocess=True)
    event.record()
    return event.ipc_handle()


def _send_store(
    client: MessageQueueClient,
    key: IPCCacheEngineKey,
    block_offset: int = 0,
    block_size: int = 16,
) -> str:
    """STORE — store KV cache blocks. Returns status string."""
    num_tokens = key.end - key.start
    num_blocks = num_tokens // block_size
    block_ids = list(range(block_offset, block_offset + num_blocks))
    payloads = [key, _INSTANCE_ID, block_ids, _make_event_handle()]
    result = _call(client, RequestType.STORE, payloads)
    if result is _TIMEOUT:
        return "timeout"
    return "stored" if result[1] else "store_failed"


def _send_retrieve(
    client: MessageQueueClient,
    key: IPCCacheEngineKey,
    chunk_size: int,
    hit_chunks: int,
    block_offset: int = 0,
    block_size: int = 16,
) -> str:
    """RETRIEVE — retrieve KV cache blocks. Returns status."""
    hit_tokens = hit_chunks * chunk_size
    num_blocks = hit_tokens // block_size
    block_ids = list(range(block_offset, block_offset + num_blocks))
    payloads = [
        key,
        _INSTANCE_ID,
        block_ids,
        _make_event_handle(),
        0,  # skip_first_n_tokens
    ]
    result = _call(client, RequestType.RETRIEVE, payloads)
    if result is _TIMEOUT:
        return "timeout"
    return "retrieved" if result[1] else "retrieve_failed"


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
    """
    blocks = list(range(block_offset, block_offset + num_blocks))
    compressed = compress_slot_mapping(blocks)
    parts: list[str] = []
    for item in compressed:
        if isinstance(item, list):
            parts.append("[%d,%d]" % (item[0], item[1]))
        else:
            parts.append(str(item))
    block_ids = ",".join(parts)
    # The MP /kvcache/check endpoint is block-native: its
    # chunk_size counts blocks per chunk, while our caller passes
    # in the server-side token-level chunk_size. Convert here.
    if chunk_size % block_size != 0:
        print(
            "  [WARNING] chunk_size %d not a multiple of block_size %d; "
            "skipping checksum query" % (chunk_size, block_size)
        )
        return None
    chunk_size_blocks = chunk_size // block_size
    url = (
        "%s/kvcache/check?block_ids=%s&block_size=%d&chunk_size=%d&layerwise=false"
    ) % (
        http_base,
        block_ids,
        block_size,
        chunk_size_blocks,
    )
    try:
        req = urllib.request.Request(url)
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


def _process_request(
    client: MessageQueueClient,
    seq_no: int,
    num_tokens: int,
    chunk_size: int,
    pass_label: str,
    http_base: str = "",
    block_size: int = 16,
    total_blocks: int = 1024,
) -> list[str] | None:
    """Run the full lookup -> retrieve/store flow."""
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
    )

    # 1. LOOKUP
    t0 = time.monotonic()
    if not _send_lookup(client, lookup_key):
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

    # 3. RETRIEVE hit portion
    if hit_chunks > 0:
        retrieve_key = _make_key(
            token_ids,
            request_id,
            start=0,
            end=hit_tokens,
            worker_id=0,
        )
        t1 = time.monotonic()
        status = _send_retrieve(
            client,
            retrieve_key,
            chunk_size,
            hit_chunks,
            block_offset=block_offset,
            block_size=block_size,
        )
        retrieve_ms = (time.monotonic() - t1) * 1000
        print(
            "  [seq %d/%s] RETRIEVE: %s "
            "(%d tokens, %.1f ms)"
            % (
                seq_no,
                pass_label,
                status,
                hit_tokens,
                retrieve_ms,
            )
        )

    # 4. STORE miss portion
    if miss_chunks > 0:
        store_start = hit_tokens
        store_end = num_full_tokens
        store_key = _make_key(
            token_ids,
            request_id,
            start=store_start,
            end=store_end,
            worker_id=0,
        )
        t2 = time.monotonic()
        store_block_off = block_offset + (hit_tokens // block_size)
        status = _send_store(
            client,
            store_key,
            block_offset=store_block_off,
            block_size=block_size,
        )
        store_ms = (time.monotonic() - t2) * 1000
        print(
            "  [seq %d/%s] STORE: %s "
            "(%d tokens, %.1f ms)"
            % (
                seq_no,
                pass_label,
                status,
                store_end - store_start,
                store_ms,
            )
        )

    # 5. Query checksums via HTTP API
    checksums = None
    if http_base and num_full_tokens > 0:
        checksums = _query_checksum(
            http_base,
            block_offset,
            num_blocks,
            block_size,
            chunk_size,
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
    return checksums


# ------------------------------------------------------------------ #
#  Server query helper                                                 #
# ------------------------------------------------------------------ #


def _get_chunk_size(client: MessageQueueClient) -> int:
    """Query the server's chunk size."""
    result = _call(client, RequestType.GET_CHUNK_SIZE, [])
    if result is _TIMEOUT or result is None:
        return 256  # fallback
    return int(result)
