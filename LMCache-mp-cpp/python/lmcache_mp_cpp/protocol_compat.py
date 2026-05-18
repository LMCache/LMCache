# SPDX-License-Identifier: Apache-2.0
"""Python wrappers for native MP protocol constants."""

# Future
from __future__ import annotations

# Standard
import ctypes

# Third Party
from lmcache_mp_cpp.bindings import _load_library


def protocol_version() -> int:
    return int(_load_library().lmcache_mp_cpp_protocol_version())


def request_type_value(name: str) -> int | None:
    lib = _load_library()
    out = ctypes.c_uint32()
    rc = lib.lmcache_mp_cpp_request_type_value(
        name.encode("utf-8"),
        ctypes.byref(out),
    )
    if rc < 0:
        raise RuntimeError("native request type lookup failed")
    if rc == 0:
        return None
    return int(out.value)


def request_type_name(value: int) -> str | None:
    raw = _load_library().lmcache_mp_cpp_request_type_name(int(value))
    if raw is None:
        return None
    return raw.decode("utf-8")
