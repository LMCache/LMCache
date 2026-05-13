# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch ``torch.cuda`` on CPU-only platforms.

When CUDA is unavailable the helpers here replace core
``torch.cuda`` classes and functions with harmless no-op stubs
so that business-logic code does not crash with
``AssertionError``.
"""

# Standard
from typing import Any

# Third Party
import torch

HAS_CUDA: bool = torch.cuda.is_available()

_cuda_compat_installed: bool = False


class _MockCudaStream:
    """Minimal stand-in for ``torch.cuda.Stream``."""

    def __init__(
        self,
        device: Any = None,  # noqa: ARG002
        priority: int = 0,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait_event(
        self,
        event: Any,  # noqa: ARG002
    ) -> None:
        pass

    def record_event(
        self,
        event: Any = None,  # noqa: ARG002
    ) -> "_MockCudaEvent":
        return _MockCudaEvent()

    @property
    def cuda_stream(self) -> int:
        return 0

    @staticmethod
    def priority_range() -> tuple[int, int]:
        return (-1, 0)


class _MockCudaEvent:
    """Minimal stand-in for ``torch.cuda.Event``."""

    def __init__(
        self,
        *,
        interprocess: bool = False,  # noqa: ARG002
        enable_timing: bool = False,  # noqa: ARG002
    ) -> None:
        pass

    def record(
        self,
        stream: Any = None,  # noqa: ARG002
    ) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait(
        self,
        stream: Any = None,  # noqa: ARG002
    ) -> None:
        pass

    def query(self) -> bool:
        return True

    def ipc_handle(self) -> bytes:
        return b""

    @classmethod
    def from_ipc_handle(
        cls,
        device: Any,  # noqa: ARG003
        handle: bytes,  # noqa: ARG003
    ) -> "_MockCudaEvent":
        return cls()


class _FakeCudart:
    """No-op replacement for ``torch.cuda.cudart()``."""

    @staticmethod
    def cudaHostRegister(
        ptr: int,  # noqa: ARG004, N802
        size: int,  # noqa: ARG004
        flags: int,  # noqa: ARG004
    ) -> None:
        return None

    @staticmethod
    def cudaHostUnregister(
        ptr: int,  # noqa: ARG004, N802
    ) -> None:
        return None


class _FakeDeviceProperties:
    """No-op replacement for device properties."""

    total_memory: int = 0
    name: str = "CPU (no CUDA)"
    major: int = 0
    minor: int = 0
    uuid: str = "cpu-0000"


class _NoopCtx:
    """No-op context manager / type stub.

    Unlike a ``@contextmanager`` function, a *class* supports
    the ``|`` union operator used in runtime type annotations
    such as ``torch.cuda.device | None``.
    """

    def __init__(
        self,
        *args: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:  # noqa: ARG002
        pass


def install_cuda_compat() -> None:
    """Monkey-patch ``torch.cuda`` when CUDA is unavailable.

    Must be called exactly once, at platform package init time.

    The patch is restricted to hosts where the detected accelerator is
    the CPU fallback so accelerator-specific code paths (``xpu``,
    ``hpu``, ...) keep their authentic ``torch.cuda`` shape — see
    :doc:`docs/design/ARCHITECTURE_MULTI_HARDWARE` for the per-layer
    contract.
    """
    global _cuda_compat_installed  # noqa: PLW0603
    if _cuda_compat_installed or HAS_CUDA:
        return
    # First Party
    from lmcache import torch_device_type

    if torch_device_type != "cpu":
        return
    _cuda_compat_installed = True

    cuda = torch.cuda

    # Classes
    cuda.Stream = _MockCudaStream  # type: ignore[misc,assignment]
    cuda.Event = _MockCudaEvent  # type: ignore[misc,assignment]

    # Context managers
    cuda.device = _NoopCtx  # type: ignore[misc,assignment]
    cuda.stream = _NoopCtx  # type: ignore[assignment]

    # Functions
    cuda.current_device = lambda: 0  # type: ignore[assignment]
    cuda.device_count = lambda: 0  # type: ignore[assignment]
    cuda.synchronize = (  # type: ignore[assignment]
        lambda device=None: None  # noqa: ARG005
    )
    cuda.empty_cache = lambda: None  # type: ignore[assignment]
    cuda.init = lambda: None  # type: ignore[assignment]
    cuda.set_device = (  # type: ignore[assignment]
        lambda device: None  # noqa: ARG005
    )
    cuda.cudart = (  # type: ignore[assignment]
        lambda: _FakeCudart()
    )

    def _fake_props(
        device: Any = 0,  # noqa: ARG001
    ) -> _FakeDeviceProperties:
        return _FakeDeviceProperties()

    cuda.get_device_properties = (  # type: ignore[assignment]
        _fake_props
    )
    cuda._lazy_init = lambda: None  # type: ignore[attr-defined]
