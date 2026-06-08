# SPDX-License-Identifier: Apache-2.0
"""Test data construction helpers for L2 adapter benchmarks."""

# Future
from __future__ import annotations

# Standard
import select

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

_KB = 1024
_DEFAULT_L1_ALIGN_BYTES = 4096


def _round_up(value: int, align_bytes: int) -> int:
    return (value + align_bytes - 1) & ~(align_bytes - 1)


def _validate_align_bytes(align_bytes: int) -> None:
    if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
        raise ValueError("align_bytes must be a positive power of two")


def make_aligned_l1_buffer(
    size: int,
    align_bytes: int = _DEFAULT_L1_ALIGN_BYTES,
) -> torch.Tensor:
    """Create a CPU uint8 tensor whose data pointer satisfies *align_bytes*.

    Args:
        size: Requested logical capacity in bytes.
        align_bytes: Required byte alignment for the returned tensor.

    Returns:
        A contiguous ``torch.uint8`` CPU tensor with capacity rounded up to
        ``align_bytes`` and a data pointer aligned to ``align_bytes``.

    Raises:
        ValueError: If ``size`` is negative or ``align_bytes`` is not a
            positive power of two.
    """
    if size < 0:
        raise ValueError("size must be >= 0")
    _validate_align_bytes(align_bytes)
    if size == 0:
        return torch.empty(0, dtype=torch.uint8)

    aligned_size = _round_up(size, align_bytes)
    base_buffer = torch.empty(aligned_size + align_bytes, dtype=torch.uint8)
    offset = (-base_buffer.data_ptr()) % align_bytes
    return base_buffer[offset : offset + aligned_size]


def make_object_keys(
    num_keys: int, model_name: str = "bench-model", key_offset: int = 0
) -> list[ObjectKey]:
    """Generate *num_keys* unique ``ObjectKey`` instances for benchmarking.

    ``ObjectKey`` is a frozen dataclass with field order:
    (chunk_hash, model_name, kv_rank).

    Args:
        num_keys: Number of keys to generate.
        model_name: Model name embedded in each key.
        key_offset: Starting index offset to ensure uniqueness across threads.
    """
    keys: list[ObjectKey] = []
    for i in range(num_keys):
        idx = key_offset + i
        # chunk_hash: 16 bytes derived from index to guarantee uniqueness
        chunk_hash = idx.to_bytes(16, "big")
        keys.append(
            ObjectKey(
                chunk_hash=chunk_hash,
                model_name=model_name,
                kv_rank=idx,
            )
        )
    return keys


def make_memory_objects(
    num_keys: int,
    data_size: int,
) -> list[MemoryObj]:
    """Create *num_keys* ``TensorMemoryObj`` instances of *data_size* bytes.

    Each returned object owns an independent ``data_size``-byte tensor
    pre-filled with a distinguishing byte pattern (key index mod 256)
    so that ``verify_round_trip`` can detect cross-key corruption after
    a store -> load cycle.

    Per-call memory: ``num_keys * data_size``.
    """
    # Independent buffers with distinguishing fill patterns for verify.
    objects: list[MemoryObj] = []
    for i in range(num_keys):
        raw_tensor = torch.empty(data_size, dtype=torch.uint8)
        raw_tensor.fill_(i & 0xFF)
        metadata = MemoryObjMetadata(
            shape=torch.Size([data_size]),
            dtype=torch.uint8,
            address=raw_tensor.data_ptr(),
            phy_size=data_size * raw_tensor.element_size(),
            fmt=MemoryFormat.KV_2LTD,
            ref_count=1,
        )
        objects.append(
            TensorMemoryObj(
                raw_data=raw_tensor,
                metadata=metadata,
                parent_allocator=None,
            )
        )
    return objects


def create_l1_memory_desc(
    buffer: torch.Tensor,
    align_bytes: int = _DEFAULT_L1_ALIGN_BYTES,
) -> L1MemoryDesc:
    """Create an L1 memory descriptor for a contiguous test buffer.

    Args:
        buffer: Contiguous tensor backing the benchmark's synthetic L1 memory.
        align_bytes: Byte alignment advertised to L2 adapters.

    Returns:
        An ``L1MemoryDesc`` describing the supplied buffer.

    Raises:
        ValueError: If ``align_bytes`` is not a positive power of two.
    """
    _validate_align_bytes(align_bytes)
    flat_buffer = buffer.view(-1)
    return L1MemoryDesc(
        ptr=flat_buffer.data_ptr(),
        size=flat_buffer.numel() * flat_buffer.element_size(),
        align_bytes=align_bytes,
    )


def wait_eventfd(efd: int, timeout: float = 60.0) -> bool:
    """Block until the eventfd is signalled or *timeout* seconds elapse.

    Uses ``select.poll`` + ``consume_fd`` for cross-platform compatibility.

    Returns True if the fd was signalled, False on timeout.
    """
    poller = select.poll()
    poller.register(efd, select.POLLIN)
    # poll() expects timeout in milliseconds
    events = poller.poll(timeout * 1000)
    if events:
        consume_fd(efd)
        return True
    return False


def verify_round_trip(keys, store_objects, load_objects, log) -> bool:
    """Verify that loaded data matches what was stored.

    Compares the underlying ``raw_data`` tensors directly (more efficient
    than converting via ``byte_array``).
    """
    mismatches = 0
    for i, (s_obj, l_obj) in enumerate(zip(store_objects, load_objects, strict=True)):
        if not torch.equal(s_obj.raw_data, l_obj.raw_data):
            mismatches += 1
            log(
                f"  [Verify] Key {i}: MISMATCH "
                f"(store {s_obj.get_physical_size()} bytes "
                f"vs load {l_obj.get_physical_size()} bytes)"
            )
    if mismatches == 0:
        log(f"  [Verify] All {len(keys)} keys data verified OK.")
        return True
    log(f"  [Verify] {mismatches}/{len(keys)} keys have data mismatches!")
    return False
