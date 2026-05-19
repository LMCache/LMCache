# SPDX-License-Identifier: Apache-2.0
"""Python wrappers for native MP key/hash compatibility helpers."""

# Future
from __future__ import annotations

# Standard
from typing import Iterable
import ctypes

# Third Party
from lmcache_mp_cpp.bindings import _load_library

BLAKE3_OUT_LEN = 32
_MAX_KV_RANK_FIELD = 255


def _token_array(tokens: Iterable[int]) -> tuple[ctypes.Array[ctypes.c_uint32], int]:
    values = [int(token) for token in tokens]
    arr = (ctypes.c_uint32 * len(values))(*values)
    return arr, len(values)


def _out32() -> ctypes.Array[ctypes.c_uint8]:
    return (ctypes.c_uint8 * BLAKE3_OUT_LEN)()


def blake3_none_hash() -> bytes:
    lib = _load_library()
    out = _out32()
    rc = lib.lmcache_mp_cpp_blake3_none_hash(out)
    if rc != 1:
        raise RuntimeError("native blake3 none hash failed")
    return bytes(out)


def blake3_hash_tokens(
    tokens: Iterable[int], prefix_hash: bytes | None = None
) -> bytes:
    lib = _load_library()
    prefix = prefix_hash if prefix_hash is not None else blake3_none_hash()
    prefix_arr = (ctypes.c_uint8 * len(prefix)).from_buffer_copy(prefix)
    token_arr, token_count = _token_array(tokens)
    out = _out32()
    rc = lib.lmcache_mp_cpp_blake3_hash_tokens(
        prefix_arr,
        len(prefix),
        token_arr,
        token_count,
        out,
    )
    if rc != 1:
        raise RuntimeError(f"native blake3 hash failed with rc={rc}")
    return bytes(out)


def compute_kv_rank(
    world_size: int,
    global_rank: int,
    local_world_size: int,
    local_rank: int,
) -> int:
    lib = _load_library()
    fields = {
        "world_size": int(world_size),
        "global_rank": int(global_rank),
        "local_world_size": int(local_world_size),
        "local_rank": int(local_rank),
    }
    for name, value in fields.items():
        if value < 0 or value > _MAX_KV_RANK_FIELD:
            raise ValueError(f"{name} must fit in one byte, got {value}")
    return int(
        lib.lmcache_mp_cpp_compute_kv_rank(
            fields["world_size"],
            fields["global_rank"],
            fields["local_world_size"],
            fields["local_rank"],
        )
    )


def expand_kv_ranks(world_size: int, worker_id: int | None) -> list[int]:
    lib = _load_library()
    max_ranks = int(world_size) if worker_id is None else 1
    out = (ctypes.c_uint32 * max_ranks)()
    out_count = ctypes.c_uint64()
    rc = lib.lmcache_mp_cpp_expand_kv_ranks(
        int(world_size),
        -1 if worker_id is None else int(worker_id),
        out,
        max_ranks,
        ctypes.byref(out_count),
    )
    if rc != 1:
        raise RuntimeError(f"native KV-rank expansion failed with rc={rc}")
    return [int(out[i]) for i in range(out_count.value)]


def object_key_string(
    model_name: str,
    kv_rank: int,
    chunk_hash: bytes,
    cache_salt: str = "",
) -> str:
    lib = _load_library()
    hash_arr = (ctypes.c_uint8 * len(chunk_hash)).from_buffer_copy(chunk_hash)
    needed = ctypes.c_uint64()
    out = ctypes.create_string_buffer(
        len(model_name.encode("utf-8"))
        + len(chunk_hash) * 2
        + len(cache_salt.encode("utf-8"))
        + 32
    )
    rc = lib.lmcache_mp_cpp_object_key_string(
        model_name.encode("utf-8"),
        int(kv_rank),
        hash_arr,
        len(chunk_hash),
        cache_salt.encode("utf-8"),
        out,
        len(out),
        ctypes.byref(needed),
    )
    if rc != 1:
        raise RuntimeError(f"native ObjectKey string failed with rc={rc}")
    return out.value.decode("utf-8")
