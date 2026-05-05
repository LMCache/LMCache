# SPDX-License-Identifier: Apache-2.0

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any, cast
import importlib
import select
import sys
import types

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd
from lmcache.v1.storage_backend.raw_block import RawBlockCoreConfig

RAW_BLOCK_CI_CAPACITY_BYTES = 128 * 1024 * 1024
RAW_BLOCK_CI_BLOCK_ALIGN = 4096
RAW_BLOCK_CI_HEADER_BYTES = 4096
RAW_BLOCK_CI_SLOT_BYTES = 64 * 1024
RAW_BLOCK_CI_META_TOTAL_BYTES = 1 * 1024 * 1024


def make_raw_block_file(
    tmp_path: Path,
    size_bytes: int = RAW_BLOCK_CI_CAPACITY_BYTES,
) -> Path:
    path = tmp_path / "raw_block_ci.bin"
    with open(path, "wb") as f:
        f.truncate(size_bytes)
    return path


def make_raw_block_core_config(
    path: Path,
    capacity_bytes: int = RAW_BLOCK_CI_CAPACITY_BYTES,
) -> RawBlockCoreConfig:
    return RawBlockCoreConfig(
        device_path=str(path),
        capacity_bytes=capacity_bytes,
        block_align=RAW_BLOCK_CI_BLOCK_ALIGN,
        header_bytes=RAW_BLOCK_CI_HEADER_BYTES,
        slot_bytes=RAW_BLOCK_CI_SLOT_BYTES,
        use_odirect=False,
        enable_zero_copy=False,
        meta_total_bytes=RAW_BLOCK_CI_META_TOTAL_BYTES,
        meta_magic=b"LMCIDX01",
        meta_version=1,
        meta_checkpoint_interval_sec=60,
        meta_idle_quiet_ms=0,
        meta_enable_periodic=False,
        meta_verify_on_load=True,
        io_engine="posix",
        iouring_queue_depth=8,
    )


def make_object_key(chunk_id: int, model_name: str = "raw_block_ci") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def make_memory_obj(payload: bytes | bytearray | memoryview) -> TensorMemoryObj:
    data = bytes(payload)
    raw_data = torch.tensor(list(data), dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size([len(data)]),
        dtype=torch.uint8,
        address=0,
        phy_size=len(data),
        fmt=MemoryFormat.BINARY,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def make_empty_memory_obj(size_bytes: int) -> TensorMemoryObj:
    raw_data = torch.zeros(size_bytes, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size_bytes]),
        dtype=torch.uint8,
        address=0,
        phy_size=size_bytes,
        fmt=MemoryFormat.BINARY,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def memory_obj_bytes(obj: TensorMemoryObj) -> bytes:
    return bytes(obj.byte_array)


def wait_for_event_fd(event_fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if not events:
        return False
    try:
        consume_fd(event_fd)
    except BlockingIOError:
        pass
    return True


def install_native_storage_ops_fallback() -> None:
    try:
        native_storage_ops = importlib.import_module("lmcache.native_storage_ops")
        if hasattr(native_storage_ops, "Bitmap") and hasattr(
            native_storage_ops, "TTLLock"
        ):
            return
    except Exception:
        pass

    class Bitmap:
        def __init__(self, size: int, first_n: int = 0) -> None:
            self._size = int(size)
            self._bits = {i for i in range(min(int(first_n), self._size))}

        def set(self, index: int) -> None:
            index = int(index)
            if index < 0 or index >= self._size:
                raise IndexError(index)
            self._bits.add(index)

        def test(self, index: int) -> bool:
            return int(index) in self._bits

        def get_indices_list(self) -> list[int]:
            return sorted(self._bits)

        def popcount(self) -> int:
            return len(self._bits)

        def count_leading_ones(self) -> int:
            count = 0
            while count in self._bits:
                count += 1
            return count

        def gather(self, values):
            return [values[i] for i in self.get_indices_list()]

        def __and__(self, other: "Bitmap") -> "Bitmap":
            size = min(self._size, other._size)
            result = Bitmap(size)
            result._bits = {i for i in self._bits & other._bits if i < size}
            return result

        def __iand__(self, other: "Bitmap") -> "Bitmap":
            self._bits &= other._bits
            self._bits = {i for i in self._bits if i < self._size}
            return self

        def __or__(self, other: "Bitmap") -> "Bitmap":
            size = max(self._size, other._size)
            result = Bitmap(size)
            result._bits = set(self._bits | other._bits)
            return result

        def __ior__(self, other: "Bitmap") -> "Bitmap":
            self._size = max(self._size, other._size)
            self._bits |= other._bits
            return self

        def __invert__(self) -> "Bitmap":
            result = Bitmap(self._size)
            result._bits = set(range(self._size)) - self._bits
            return result

        def __str__(self) -> str:
            return "".join("1" if i in self._bits else "0" for i in range(self._size))

    class TTLLock:
        pass

    fallback_module = types.ModuleType("lmcache.native_storage_ops")
    fallback_module_any = cast(Any, fallback_module)
    fallback_module_any.Bitmap = Bitmap
    fallback_module_any.TTLLock = TTLLock
    sys.modules["lmcache.native_storage_ops"] = fallback_module
