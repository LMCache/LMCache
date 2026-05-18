# SPDX-License-Identifier: Apache-2.0
"""Python wrappers for native MP IPCCacheEngineKey decoding."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import ctypes

# Third Party
from lmcache_mp_cpp.bindings import _CIpcKeySummary, _load_library, _readonly_ptr


@dataclass(frozen=True)
class DecodedIpcCacheEngineKey:
    model_name: str
    world_size: int
    worker_id: int | None
    token_ids: tuple[int, ...]
    start: int
    end: int
    request_id: str
    cache_salt: str


def _summary(encoded: bytes) -> _CIpcKeySummary:
    lib = _load_library()
    arr, size = _readonly_ptr(encoded)
    out = _CIpcKeySummary()
    rc = lib.lmcache_mp_cpp_ipc_key_summary(arr, size, ctypes.byref(out))
    if rc != 1:
        raise RuntimeError(f"native IPCCacheEngineKey summary failed with rc={rc}")
    return out


def decode_ipc_key(encoded: bytes) -> DecodedIpcCacheEngineKey:
    lib = _load_library()
    first = _summary(encoded)
    arr, size = _readonly_ptr(encoded)
    out = _CIpcKeySummary()
    model_name = ctypes.create_string_buffer(first.model_name_len + 1)
    request_id = ctypes.create_string_buffer(first.request_id_len + 1)
    cache_salt = ctypes.create_string_buffer(first.cache_salt_len + 1)
    token_count = int(first.token_count)
    tokens = (ctypes.c_uint32 * max(1, token_count))()

    rc = lib.lmcache_mp_cpp_decode_ipc_key(
        arr,
        size,
        ctypes.byref(out),
        model_name,
        len(model_name),
        request_id,
        len(request_id),
        cache_salt,
        len(cache_salt),
        tokens,
        token_count,
    )
    if rc != 1:
        raise RuntimeError(f"native IPCCacheEngineKey decode failed with rc={rc}")

    worker_id = None if out.worker_id < 0 else int(out.worker_id)
    return DecodedIpcCacheEngineKey(
        model_name=model_name.value.decode("utf-8"),
        world_size=int(out.world_size),
        worker_id=worker_id,
        token_ids=tuple(int(tokens[i]) for i in range(out.token_count)),
        start=int(out.start),
        end=int(out.end),
        request_id=request_id.value.decode("utf-8"),
        cache_salt=cache_salt.value.decode("utf-8"),
    )


def object_key_strings(
    encoded: bytes,
    *,
    chunk_size: int,
    start: int | None = None,
    end: int | None = None,
) -> list[str]:
    summary = _summary(encoded)
    object_start = int(summary.start if start is None else start)
    object_end = int(summary.end if end is None else end)
    lib = _load_library()
    arr, size = _readonly_ptr(encoded)
    needed = ctypes.c_uint64()
    count = ctypes.c_uint64()
    dummy = ctypes.create_string_buffer(1)
    rc = lib.lmcache_mp_cpp_ipc_key_object_key_strings(
        arr,
        size,
        int(chunk_size),
        object_start,
        object_end,
        dummy,
        0,
        ctypes.byref(needed),
        ctypes.byref(count),
    )
    if rc not in (1, -2):
        raise RuntimeError(f"native ObjectKey expansion failed with rc={rc}")
    if count.value == 0:
        return []

    out = ctypes.create_string_buffer(needed.value)
    rc = lib.lmcache_mp_cpp_ipc_key_object_key_strings(
        arr,
        size,
        int(chunk_size),
        object_start,
        object_end,
        out,
        len(out),
        ctypes.byref(needed),
        ctypes.byref(count),
    )
    if rc != 1:
        raise RuntimeError(f"native ObjectKey expansion failed with rc={rc}")
    return [
        item.decode("utf-8")
        for item in bytes(out.raw[: needed.value]).rstrip(b"\0").split(b"\0")
    ]
