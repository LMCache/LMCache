# SPDX-License-Identifier: Apache-2.0
"""Cross-platform ``ExternalStream`` abstraction.

Defines the :class:`ExternalStreamLike` protocol every backend must
satisfy and a single :func:`make_external_stream` factory that picks
a concrete implementation based on the running host.

Backend implementations live under ``platform/<device>/stream.py`` so
each accelerator can evolve independently without touching this file
or sibling backends:

* ``cuda/stream.py`` -> cupy-backed ``ExternalStream``
* ``cpu/stream.py``  -> pure-Python mock fallback
* ``xpu/stream.py``, ``hpu/stream.py`` -> reserved for future devices

This module never imports device-specific bindings at import time;
each backend is loaded lazily inside the factory so CPU-only hosts
never touch ``cuda/`` (and vice-versa).
"""

# Future
from __future__ import annotations

# Standard
from typing import Any, Callable, Protocol

# Third Party
import torch

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class ExternalStreamLike(Protocol):
    """Structural type describing the subset of the cupy stream API we use."""

    ptr: int

    def launch_host_func(self, func: Callable[[Any], None], arg: Any) -> None: ...


def _extract_raw_ptr(torch_stream: torch.cuda.Stream | None) -> int:
    """Best-effort extraction of a CUDA stream handle from a torch stream.

    ``cuda_stream`` only exists on a real CUDA-backed ``torch.cuda.Stream``.
    On CPU-only hosts the attribute may be missing or raise when accessed,
    so guard it and fall back to ``0`` — the mock stream treats that as
    "no usable handle" and synthesizes a fake non-zero id.
    """
    try:
        return int(torch_stream.cuda_stream) if torch_stream is not None else 0
    except Exception:  # pragma: no cover - platform dependent
        return 0


def make_external_stream(
    torch_stream: torch.cuda.Stream, device_index: int
) -> ExternalStreamLike:
    """Build an external stream wrapper around a ``torch.cuda.Stream``.

    Dispatch order:

    1. On CUDA-capable hosts, try :func:`cuda.stream.make_cuda_external_stream`
       (cupy-backed).  A ``None`` return means cupy is not importable and
       we continue down the chain.
    2. Otherwise (or when CUDA backends declined), fall back to the CPU
       mock in :mod:`cpu.stream`.

    New accelerators can be plugged in by adding another branch here or,
    later, by wiring a Registry — callers never need to change.
    """
    raw_ptr = _extract_raw_ptr(torch_stream)

    if torch.cuda.is_available():
        # Local import keeps CPU-only hosts from pulling in cupy.
        # First Party
        from lmcache.v1.platform.cuda.stream import make_cuda_external_stream

        stream = make_cuda_external_stream(raw_ptr, device_index)
        if stream is not None:
            return stream

    # First Party
    from lmcache.v1.platform.cpu.stream import MockExternalStream

    logger.info(f"make_external_stream: MockExternalStream {raw_ptr}")
    return MockExternalStream(raw_ptr)
