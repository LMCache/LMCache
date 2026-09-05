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
    from lmcache.v1.kv_layer_groups import (
        DTYPE_MAP,
        KVLayerGroupInfo,
    )
    from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
    from lmcache.v1.multiprocess.futures import MessagingFuture
    from lmcache.v1.multiprocess.transport.base import RequestClient
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

# TP > 1 support: each simulated worker registers under a distinct
# ``instance_id`` so the server can hold one context per rank. Kept
# well above legacy ``_INSTANCE_ID = 0`` so single-worker bench runs
# and the multi-worker path never collide on the server side.
_INSTANCE_ID_BASE = 1000

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


def _wait_for_result(
    future: MessagingFuture[Any],
    timeout_s: float = _DEFAULT_RPC_TIMEOUT_S,
) -> Any:
    """Wait for an RPC future and convert a timeout to ``_TIMEOUT``.

    Returns the decoded response (possibly ``None`` for void replies)
    on success, or the sentinel ``_TIMEOUT`` on RPC timeout.
    """
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
#  KV cache allocation                                                 #
# ------------------------------------------------------------------ #


# The server's vLLM detector identifies MLA layers by tensor rank: each
# layer must be rank-3 ``(NB, BS, HS)`` (see ``VLLM_Detector.discover``
# in ``lmcache/v1/gpu_connector/kv_format/detectors/vllm.py``). Classical
# split-K/V is rank-5 ``(2, NB, BS, NH, HS)``. Sharing this shape recipe
# across allocation and checksum helpers keeps the bench in sync with the
# detector contract regardless of transfer mode.
def _is_mla_kv_size(kv_size: int) -> bool:
    """``kv_size == 1`` marks a single-plane KV group (MLA / fused-K/V).

    Single source of truth for "is this group MLA?". Derived helpers
    (:func:`_make_alloc_shape`, :func:`_tensor_is_mla`) express the same
    contract in shape-space so allocation and checksum paths do not diverge.
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
    """Allocate paged KV cache tensors on the selected device.

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


def _send_lookup(
    client: RequestClient,
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
    result = _wait_for_result(client.lookup(key, tp_size))
    return result is not _TIMEOUT


def _poll_prefetch_status(
    client: RequestClient,
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
        result = _wait_for_result(client.query_prefetch_status(request_id))
        if result is _TIMEOUT:
            # RPC timeout — treat as giving up on this poll cycle.
            return None
        if result is not None:
            return result
        time.sleep(poll_interval)
    return None


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


def _send_end_session(
    client: RequestClient,
    request_id: str,
) -> None:
    """END_SESSION — clean up server-side session state."""
    _wait_for_result(client.end_session(request_id))


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
#  Server query helper                                                 #
# ------------------------------------------------------------------ #


def _get_chunk_size(client: RequestClient) -> int:
    """Query the server's chunk size."""
    result = _wait_for_result(client.get_chunk_size())
    if result is _TIMEOUT or result is None:
        return 256  # fallback
    return int(result)
