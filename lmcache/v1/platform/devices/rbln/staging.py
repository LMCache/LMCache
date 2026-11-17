# SPDX-License-Identifier: Apache-2.0
"""Device staging for RBLN HND restores.

RBLN device DRAM is one pool per chiplet. An unplaced allocation lands on
chiplet 0 next to that chiplet's share of the model, so staging rows are
split into one shard per chiplet when placement is available.

The restore set holds head-major and token-major buffers per chiplet shard.
Placement spreads the rows over chiplets when its APIs are available; the
unplaced path reserves one pair on chiplet 0. The first HND transfer in
either direction reserves this memory, before a store can publish chunks.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

_CHIPLET_COUNTS: dict[str, int] = {}
_CHIPLET_COUNTS_LOCK = threading.Lock()


@dataclass(frozen=True)
class HndStaging:
    """Restore staging of one geometry, placed per chiplet when supported.

    Attributes:
        head_major: Per-shard ``[rows, NH, BS, HS]`` buffers.
        token_major: Per-shard ``[rows, BS, NH, HS]`` buffers of the same size.
        lock: Held by the caller for the whole transfer using this set.
    """

    head_major: list[torch.Tensor]
    token_major: list[torch.Tensor]
    lock: threading.Lock = field(default_factory=threading.Lock)


_Geometry = tuple[str, torch.dtype, int, int, int, int, int]
_GEOMETRY: _Geometry | None = None
_STAGING: HndStaging | None = None
_STAGING_LOCK = threading.Lock()
_PLACEMENT_WARNING_LOGGED = False


def placement_supported() -> bool:
    """Whether torch-rbln provides the optional device-placement APIs.

    Returns:
        True when ``set_device_alloc`` and ``DevicePlacement`` are available
        on ``torch.rbln``.
    """
    rbln = getattr(torch, "rbln", None)
    return rbln is not None and all(
        callable(getattr(rbln, name, None))
        for name in ("set_device_alloc", "DevicePlacement")
    )


def chiplet_count(device: torch.device) -> int:
    """Chiplets of the first NPU behind ``device``.

    The count is read once per device and kept: it cannot change while the
    process runs. Host tensors,
    which tests and reference computations pass, have no chiplets to spread
    over and count as one.

    Args:
        device: Device of the paged KV cache.

    Returns:
        How many chiplets the device reports; 1 on the host.

    Raises:
        RuntimeError: If the device reports no chiplet.
    """
    if device.type == "cpu":
        return 1
    key = str(device)
    with _CHIPLET_COUNTS_LOCK:
        count = _CHIPLET_COUNTS.get(key)
        if count is None:
            properties = torch.rbln.get_device_properties(device)
            count = getattr(properties, "num_chiplet", 0)
            if count <= 0:
                raise RuntimeError(
                    f"RBLN HND transfer requires {device} device properties "
                    "with a positive num_chiplet"
                )
            _CHIPLET_COUNTS[key] = count
    return count


def shard_bounds(rows: int, shards: int) -> list[tuple[int, int]]:
    """Cut ``rows`` consecutive rows into ``shards`` near-equal ranges.

    Args:
        rows: Rows to cut.
        shards: Number of ranges.

    Returns:
        ``[lo, hi)`` row ranges, in order.
    """
    return [(rows * s // shards, rows * (s + 1) // shards) for s in range(shards)]


def hnd_staging(
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int,
    num_heads: int,
    block_size: int,
    head_size: int,
) -> HndStaging:
    """Return the shared restore staging for one HND block.

    The ``2 * num_layers`` rows are split over ``min(chiplets, num_layers)``
    shards when placement is available. Otherwise one unplaced shard is
    reserved without a chiplet query, and the process warns once. The first
    HND transfer in either direction reserves physical memory so allocation
    failures occur before a store publishes chunks. Allocation does not copy
    or compile. Only the latest geometry is retained; callers use one loaded
    model geometry and hold the set's lock for the whole restore call.

    Args:
        device: Device of the paged KV cache.
        dtype: KV dtype.
        num_layers: Layers in one block.
        num_heads: KV heads.
        block_size: Tokens per block.
        head_size: Elements per head.

    Returns:
        One restore set with a head/token buffer pair per shard.

    Raises:
        RuntimeError: If reservation or placement fails. The failed set is
            not cached, so the next transfer retries allocation.
    """
    global _GEOMETRY, _STAGING, _PLACEMENT_WARNING_LOGGED
    with _STAGING_LOCK:
        if placement_supported():
            shards = min(chiplet_count(device), num_layers)
        else:
            shards = 1
            if not _PLACEMENT_WARNING_LOGGED:
                missing = ", ".join(
                    f"torch.rbln.{name}"
                    for name in ("set_device_alloc", "DevicePlacement")
                    if not callable(getattr(getattr(torch, "rbln", None), name, None))
                )
                logger.warning(
                    "RBLN HND staging is not spread over chiplets: missing %s",
                    missing,
                )
                _PLACEMENT_WARNING_LOGGED = True
        key = (str(device), dtype, num_layers, num_heads, block_size, head_size, shards)
        if _GEOMETRY != key or _STAGING is None:
            restored = _allocate(
                device, dtype, num_layers, num_heads, block_size, head_size, shards
            )
            _GEOMETRY, _STAGING = key, restored
        return _STAGING


def _allocate(
    device: torch.device,
    dtype: torch.dtype,
    num_layers: int,
    num_heads: int,
    block_size: int,
    head_size: int,
    shards: int,
) -> HndStaging:
    head_major, token_major = [], []
    for chiplet, (lo, hi) in enumerate(shard_bounds(2 * num_layers, shards)):
        pair = (
            torch.empty(
                (hi - lo, num_heads, block_size, head_size), dtype=dtype, device=device
            ),
            torch.empty(
                (hi - lo, block_size, num_heads, head_size), dtype=dtype, device=device
            ),
        )
        for buffer in pair:
            if placement_supported() and (device.type != "cpu" or shards > 1):
                torch.rbln.set_device_alloc(
                    buffer,
                    [torch.rbln.DevicePlacement(0, chiplet, ((0, 0, hi - lo),))],
                )
            elif device.type != "cpu":
                # Unplaced device tensors are lazy until explicitly reserved.
                torch.rbln.bind_device_memory(buffer)
        head_major.append(pair[0])
        token_major.append(pair[1])
    return HndStaging(head_major, token_major)
