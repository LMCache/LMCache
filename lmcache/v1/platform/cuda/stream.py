# SPDX-License-Identifier: Apache-2.0
"""CUDA-backed ``ExternalStreamLike`` implementation.

The LMCache multiprocess server only needs one capability from ``cupy``:
the ability to attach a Python host callback to a CUDA stream via
``launch_host_func`` (ultimately ``cudaLaunchHostFunc``).  ``torch``'s
native ``torch.cuda.Stream`` does not expose this API, which is why the
code base historically imported ``cupy``.

This module isolates that dependency: callers never import ``cupy``
directly.  The dispatcher in :mod:`lmcache.v1.platform.stream` asks for
a CUDA stream via :func:`make_cuda_external_stream`; a ``None`` return
signals the caller to fall back to another backend (e.g. the mock in
:mod:`lmcache.v1.platform.cpu.stream`).

Swapping the underlying CUDA binding in the future (e.g. a ``ctypes``
wrapper around ``cudaLaunchHostFunc``) only requires editing this
file — consumers stay oblivious.
"""

# Future
from __future__ import annotations

# Standard
from typing import Any

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def _try_import_cupy() -> Any | None:
    """Import ``cupy`` lazily; return ``None`` if unavailable."""
    try:
        # Third Party
        import cupy  # type: ignore[import-not-found]

        return cupy
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.debug("cupy not available, using mock stream: %s", exc)
        return None


def make_cuda_external_stream(raw_ptr: int, device_index: int) -> Any | None:
    """Build a ``cupy.cuda.ExternalStream`` if cupy imports successfully.

    Returns ``None`` when cupy is not available so the dispatcher can
    pick another backend.
    """
    cupy = _try_import_cupy()
    if cupy is None:
        return None
    return cupy.cuda.ExternalStream(raw_ptr, device_index)
