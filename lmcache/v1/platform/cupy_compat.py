# SPDX-License-Identifier: Apache-2.0
"""Inject a fake ``cupy`` package on CPU-only platforms.

On CPU-only platforms ``cupy`` cannot be installed at all,
so we create a minimal shim module in ``sys.modules`` so that
``import cupy`` succeeds and the handful of APIs used by the
codebase (``cupy.cuda.ExternalStream``) resolve to harmless
no-ops.
"""

# Standard
from typing import Any
import sys
import types

# Third Party
import torch

HAS_CUDA: bool = torch.cuda.is_available()

_cupy_compat_installed: bool = False


class _MockCupyStream:
    """Minimal stand-in for ``cupy.cuda.ExternalStream``."""

    def __init__(
        self,
        ptr: int = 0,
        device_id: int = 0,
    ) -> None:
        self.ptr = ptr
        self.device_id = device_id

    def launch_host_func(
        self,
        fn: Any,
        *args: Any,
    ) -> None:
        fn(*args)

    def synchronize(self) -> None:
        pass


def install_cupy_compat() -> None:
    """Inject a fake ``cupy`` when CUDA is unavailable.

    Must be called exactly once, at platform package init time.

    Restricted to the CPU fallback platform so accelerator-specific
    deployments (``xpu``, ``hpu``, ...) keep whatever ``cupy``-shaped
    binding they ship without being shadowed by the no-op stub.
    """
    global _cupy_compat_installed  # noqa: PLW0603
    if _cupy_compat_installed or HAS_CUDA:
        return
    # First Party
    from lmcache import torch_device_type

    if torch_device_type != "cpu":
        return
    if "cupy" in sys.modules:
        # Real cupy already loaded — nothing to do.
        return
    _cupy_compat_installed = True

    cupy = types.ModuleType("cupy")
    cuda = types.ModuleType("cupy.cuda")
    cuda.ExternalStream = _MockCupyStream  # type: ignore[attr-defined]
    cuda.Stream = _MockCupyStream  # type: ignore[attr-defined]
    cupy.cuda = cuda  # type: ignore[attr-defined]
    sys.modules["cupy"] = cupy
    sys.modules["cupy.cuda"] = cuda
