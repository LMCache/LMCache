# SPDX-License-Identifier: Apache-2.0
"""Python wrappers for native filesystem L2 adapter helpers."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import ctypes

# Third Party
from lmcache_mp_cpp.bindings import _key_bytes, _load_library, _readonly_ptr


class FileSystemL2Adapter:
    def __init__(self, base_path: str | Path) -> None:
        self._lib = _load_library()
        self._ptr = self._lib.lmcache_mp_cpp_fs_l2_create(
            str(base_path).encode("utf-8")
        )
        if not self._ptr:
            raise RuntimeError("failed to create native filesystem L2 adapter")

    def close(self) -> None:
        if self._ptr:
            self._lib.lmcache_mp_cpp_fs_l2_destroy(self._ptr)
            self._ptr = None

    def __enter__(self) -> "FileSystemL2Adapter":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def _check(self, rc: int) -> None:
        if rc < 0:
            raise RuntimeError(self.last_error())

    def last_error(self) -> str:
        raw = self._lib.lmcache_mp_cpp_fs_l2_last_error(self._ptr)
        return raw.decode("utf-8") if raw else ""

    def put(self, key: str, data: bytes) -> None:
        arr, size = _readonly_ptr(data)
        rc = self._lib.lmcache_mp_cpp_fs_l2_put(
            self._ptr,
            _key_bytes(key),
            arr,
            size,
        )
        self._check(rc)

    def exists(self, key: str) -> bool:
        rc = self._lib.lmcache_mp_cpp_fs_l2_exists(self._ptr, _key_bytes(key))
        self._check(rc)
        return rc == 1

    def get(self, key: str) -> bytes | None:
        out_len = ctypes.c_uint64()
        rc = self._lib.lmcache_mp_cpp_fs_l2_size(
            self._ptr,
            _key_bytes(key),
            ctypes.byref(out_len),
        )
        self._check(rc)
        if rc == 0:
            return None
        out = (ctypes.c_uint8 * out_len.value)()
        rc = self._lib.lmcache_mp_cpp_fs_l2_get(
            self._ptr,
            _key_bytes(key),
            out,
            out_len.value,
        )
        self._check(rc)
        if rc == 0:
            return None
        return bytes(out)

    def delete(self, key: str) -> None:
        rc = self._lib.lmcache_mp_cpp_fs_l2_delete(self._ptr, _key_bytes(key))
        self._check(rc)

    def clear(self) -> None:
        rc = self._lib.lmcache_mp_cpp_fs_l2_clear(self._ptr)
        self._check(rc)


def fs_l2_filename(key: str) -> str:
    lib = _load_library()
    needed = ctypes.c_uint64()
    out = ctypes.create_string_buffer(len(key.encode("utf-8")) + 32)
    rc = lib.lmcache_mp_cpp_fs_l2_filename(
        key.encode("utf-8"),
        out,
        len(out),
        ctypes.byref(needed),
    )
    if rc == -2:
        out = ctypes.create_string_buffer(needed.value + 1)
        rc = lib.lmcache_mp_cpp_fs_l2_filename(
            key.encode("utf-8"),
            out,
            len(out),
            ctypes.byref(needed),
        )
    if rc != 1:
        raise RuntimeError(f"native filesystem L2 filename failed with rc={rc}")
    return out.value.decode("utf-8")
