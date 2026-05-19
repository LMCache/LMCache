# SPDX-License-Identifier: Apache-2.0
"""ctypes wrapper for the C++ tiered cache."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import ctypes

# Third Party
from lmcache_mp_cpp.build import build_library

Buffer = Union[bytes, bytearray, memoryview]


class _CStats(ctypes.Structure):
    _fields_ = [
        ("dram_bytes", ctypes.c_uint64),
        ("disk_bytes", ctypes.c_uint64),
        ("dram_entries", ctypes.c_uint64),
        ("disk_entries", ctypes.c_uint64),
        ("total_entries", ctypes.c_uint64),
        ("locked_entries", ctypes.c_uint64),
        ("lock_count", ctypes.c_uint64),
        ("locked_bytes", ctypes.c_uint64),
        ("pinned_entries", ctypes.c_uint64),
        ("eviction_count", ctypes.c_uint64),
    ]


class _CIpcKeySummary(ctypes.Structure):
    _fields_ = [
        ("token_count", ctypes.c_uint64),
        ("start", ctypes.c_uint64),
        ("end", ctypes.c_uint64),
        ("model_name_len", ctypes.c_uint64),
        ("request_id_len", ctypes.c_uint64),
        ("cache_salt_len", ctypes.c_uint64),
        ("world_size", ctypes.c_uint32),
        ("worker_id", ctypes.c_int32),
    ]


@dataclass(frozen=True)
class CacheStats:
    dram_bytes: int
    disk_bytes: int
    dram_entries: int
    disk_entries: int
    total_entries: int
    locked_entries: int
    lock_count: int
    locked_bytes: int
    pinned_entries: int
    eviction_count: int


def _load_library(path: Path | None = None) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path or build_library()))
    lib.lmcache_mp_cpp_cache_create.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
    lib.lmcache_mp_cpp_cache_create.restype = ctypes.c_void_p
    lib.lmcache_mp_cpp_cache_destroy.argtypes = [ctypes.c_void_p]
    lib.lmcache_mp_cpp_cache_destroy.restype = None
    lib.lmcache_mp_cpp_cache_put.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    lib.lmcache_mp_cpp_cache_put.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_get.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    lib.lmcache_mp_cpp_cache_get.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_exists.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lmcache_mp_cpp_cache_exists.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_size.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lmcache_mp_cpp_cache_size.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_remove.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lmcache_mp_cpp_cache_remove.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_lock.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lmcache_mp_cpp_cache_lock.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_unlock.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lmcache_mp_cpp_cache_unlock.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_pin.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lmcache_mp_cpp_cache_pin.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_unpin.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.lmcache_mp_cpp_cache_unpin.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_is_resident.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.lmcache_mp_cpp_cache_is_resident.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_clear.argtypes = [ctypes.c_void_p]
    lib.lmcache_mp_cpp_cache_clear.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_clear_force.argtypes = [ctypes.c_void_p]
    lib.lmcache_mp_cpp_cache_clear_force.restype = ctypes.c_int
    lib.lmcache_mp_cpp_cache_stats.argtypes = [ctypes.c_void_p]
    lib.lmcache_mp_cpp_cache_stats.restype = _CStats
    lib.lmcache_mp_cpp_cache_last_error.argtypes = [ctypes.c_void_p]
    lib.lmcache_mp_cpp_cache_last_error.restype = ctypes.c_char_p
    lib.lmcache_mp_cpp_cache_last_error_copy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lmcache_mp_cpp_cache_last_error_copy.restype = ctypes.c_int
    lib.lmcache_mp_cpp_blake3_none_hash.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.lmcache_mp_cpp_blake3_none_hash.restype = ctypes.c_int
    lib.lmcache_mp_cpp_blake3_hash_tokens.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint8),
    ]
    lib.lmcache_mp_cpp_blake3_hash_tokens.restype = ctypes.c_int
    lib.lmcache_mp_cpp_compute_kv_rank.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    lib.lmcache_mp_cpp_compute_kv_rank.restype = ctypes.c_uint32
    lib.lmcache_mp_cpp_expand_kv_ranks.argtypes = [
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lmcache_mp_cpp_expand_kv_ranks.restype = ctypes.c_int
    lib.lmcache_mp_cpp_object_key_string.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lmcache_mp_cpp_object_key_string.restype = ctypes.c_int
    lib.lmcache_mp_cpp_protocol_version.argtypes = []
    lib.lmcache_mp_cpp_protocol_version.restype = ctypes.c_uint32
    lib.lmcache_mp_cpp_request_type_value.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.lmcache_mp_cpp_request_type_value.restype = ctypes.c_int
    lib.lmcache_mp_cpp_request_type_name.argtypes = [ctypes.c_uint32]
    lib.lmcache_mp_cpp_request_type_name.restype = ctypes.c_char_p
    lib.lmcache_mp_cpp_ipc_key_summary.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(_CIpcKeySummary),
    ]
    lib.lmcache_mp_cpp_ipc_key_summary.restype = ctypes.c_int
    lib.lmcache_mp_cpp_decode_ipc_key.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(_CIpcKeySummary),
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
    ]
    lib.lmcache_mp_cpp_decode_ipc_key.restype = ctypes.c_int
    lib.lmcache_mp_cpp_ipc_key_object_key_strings.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lmcache_mp_cpp_ipc_key_object_key_strings.restype = ctypes.c_int
    lib.lmcache_mp_cpp_fs_l2_create.argtypes = [ctypes.c_char_p]
    lib.lmcache_mp_cpp_fs_l2_create.restype = ctypes.c_void_p
    lib.lmcache_mp_cpp_fs_l2_destroy.argtypes = [ctypes.c_void_p]
    lib.lmcache_mp_cpp_fs_l2_destroy.restype = None
    lib.lmcache_mp_cpp_fs_l2_put.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    lib.lmcache_mp_cpp_fs_l2_put.restype = ctypes.c_int
    lib.lmcache_mp_cpp_fs_l2_size.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lmcache_mp_cpp_fs_l2_size.restype = ctypes.c_int
    lib.lmcache_mp_cpp_fs_l2_get.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
    ]
    lib.lmcache_mp_cpp_fs_l2_get.restype = ctypes.c_int
    lib.lmcache_mp_cpp_fs_l2_delete.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.lmcache_mp_cpp_fs_l2_delete.restype = ctypes.c_int
    lib.lmcache_mp_cpp_fs_l2_clear.argtypes = [ctypes.c_void_p]
    lib.lmcache_mp_cpp_fs_l2_clear.restype = ctypes.c_int
    lib.lmcache_mp_cpp_fs_l2_exists.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.lmcache_mp_cpp_fs_l2_exists.restype = ctypes.c_int
    lib.lmcache_mp_cpp_fs_l2_last_error.argtypes = [ctypes.c_void_p]
    lib.lmcache_mp_cpp_fs_l2_last_error.restype = ctypes.c_char_p
    lib.lmcache_mp_cpp_fs_l2_filename.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.lmcache_mp_cpp_fs_l2_filename.restype = ctypes.c_int
    return lib


def _key_bytes(key: str) -> bytes:
    return key.encode("utf-8")


def _readonly_ptr(data: Buffer) -> tuple[ctypes.Array[ctypes.c_uint8], int]:
    view = memoryview(data).cast("B")
    if view.readonly:
        arr = (ctypes.c_uint8 * len(view)).from_buffer_copy(view)
    else:
        arr = (ctypes.c_uint8 * len(view)).from_buffer(view)
    return arr, len(view)


def _writable_ptr(data: Buffer) -> tuple[ctypes.Array[ctypes.c_uint8], int]:
    view = memoryview(data).cast("B")
    if view.readonly:
        raise TypeError("target buffer must be writable")
    arr = (ctypes.c_uint8 * len(view)).from_buffer(view)
    return arr, len(view)


def _cache_last_error(lib: ctypes.CDLL, ptr: int) -> str:
    needed = ctypes.c_uint64()
    out = ctypes.create_string_buffer(256)
    rc = lib.lmcache_mp_cpp_cache_last_error_copy(
        ptr, out, len(out), ctypes.byref(needed)
    )
    if rc == -2:
        out = ctypes.create_string_buffer(needed.value + 1)
        rc = lib.lmcache_mp_cpp_cache_last_error_copy(
            ptr, out, len(out), ctypes.byref(needed)
        )
    if rc != 1:
        return ""
    return out.value.decode("utf-8")


class TieredCache:
    """C++ DRAM/disk byte cache."""

    def __init__(
        self,
        dram_capacity_bytes: int,
        disk_path: str | Path,
        library: Path | None = None,
    ) -> None:
        self._lib = _load_library(library)
        disk = str(disk_path).encode("utf-8")
        self._ptr = self._lib.lmcache_mp_cpp_cache_create(
            int(dram_capacity_bytes), disk
        )
        if not self._ptr:
            raise RuntimeError("failed to create C++ tiered cache")

    def close(self) -> None:
        if self._ptr:
            self._lib.lmcache_mp_cpp_cache_destroy(self._ptr)
            self._ptr = None

    def __enter__(self) -> "TieredCache":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _check(self, rc: int) -> None:
        if rc < 0:
            raise RuntimeError(self.last_error())

    def last_error(self) -> str:
        return _cache_last_error(self._lib, self._ptr)

    def put(self, key: str, data: Buffer) -> None:
        arr, size = _readonly_ptr(data)
        rc = self._lib.lmcache_mp_cpp_cache_put(self._ptr, _key_bytes(key), arr, size)
        self._check(rc)

    def exists(self, key: str) -> bool:
        rc = self._lib.lmcache_mp_cpp_cache_exists(self._ptr, _key_bytes(key))
        self._check(rc)
        return rc == 1

    def size(self, key: str) -> int | None:
        out = ctypes.c_uint64()
        rc = self._lib.lmcache_mp_cpp_cache_size(
            self._ptr, _key_bytes(key), ctypes.byref(out)
        )
        self._check(rc)
        if rc == 0:
            return None
        return int(out.value)

    def get_into(self, key: str, target: Buffer) -> bool:
        arr, size = _writable_ptr(target)
        rc = self._lib.lmcache_mp_cpp_cache_get(self._ptr, _key_bytes(key), arr, size)
        self._check(rc)
        return rc == 1

    def get(self, key: str) -> bytes | None:
        size = self.size(key)
        if size is None:
            return None
        target = bytearray(size)
        if not self.get_into(key, target):
            return None
        return bytes(target)

    def remove(self, key: str) -> None:
        rc = self._lib.lmcache_mp_cpp_cache_remove(self._ptr, _key_bytes(key))
        self._check(rc)

    def lock(self, key: str) -> bool:
        rc = self._lib.lmcache_mp_cpp_cache_lock(self._ptr, _key_bytes(key))
        self._check(rc)
        return rc == 1

    def unlock(self, key: str) -> bool:
        rc = self._lib.lmcache_mp_cpp_cache_unlock(self._ptr, _key_bytes(key))
        self._check(rc)
        return rc == 1

    def pin(self, key: str) -> bool:
        rc = self._lib.lmcache_mp_cpp_cache_pin(self._ptr, _key_bytes(key))
        self._check(rc)
        return rc == 1

    def unpin(self, key: str) -> bool:
        rc = self._lib.lmcache_mp_cpp_cache_unpin(self._ptr, _key_bytes(key))
        self._check(rc)
        return rc == 1

    def is_resident(self, key: str) -> bool:
        rc = self._lib.lmcache_mp_cpp_cache_is_resident(self._ptr, _key_bytes(key))
        self._check(rc)
        return rc == 1

    def clear(self, force: bool = False) -> None:
        clear_func = (
            self._lib.lmcache_mp_cpp_cache_clear_force
            if force
            else self._lib.lmcache_mp_cpp_cache_clear
        )
        rc = clear_func(self._ptr)
        self._check(rc)

    def stats(self) -> CacheStats:
        raw = self._lib.lmcache_mp_cpp_cache_stats(self._ptr)
        return CacheStats(
            dram_bytes=int(raw.dram_bytes),
            disk_bytes=int(raw.disk_bytes),
            dram_entries=int(raw.dram_entries),
            disk_entries=int(raw.disk_entries),
            total_entries=int(raw.total_entries),
            locked_entries=int(raw.locked_entries),
            lock_count=int(raw.lock_count),
            locked_bytes=int(raw.locked_bytes),
            pinned_entries=int(raw.pinned_entries),
            eviction_count=int(raw.eviction_count),
        )
