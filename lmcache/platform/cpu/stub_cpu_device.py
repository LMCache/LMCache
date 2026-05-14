# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from contextlib import nullcontext
from typing import Any


class StubDeviceProperties:
    """Stub for torch_dev.get_device_properties() return value."""

    def __init__(self) -> None:
        self.name = "StubCPU"
        self.major = 0
        self.minor = 0
        self.total_memory = 0
        self.multi_processor_count = 0
        self.uuid = "stub-0000-0000-0000-000000000000"

    def __repr__(self) -> str:
        return f"StubDeviceProperties(name={self.name!r})"


class StubEvent:
    def __init__(
        self,
        enable_timing: bool = False,
        blocking: bool = False,
        interprocess: bool = False,
    ) -> None:
        self.enable_timing = enable_timing
        self.blocking = blocking
        self.interprocess = interprocess
        self._recorded = False
        self._handle = b"stub_ipc_handle"

    def record(self, stream: Any = None) -> None:
        self._recorded = True

    def wait(self, stream: Any = None) -> None:
        return None

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None

    def elapsed_time(self, end_event: "StubEvent") -> float:
        return 0.0

    def ipc_handle(self) -> bytes:
        return self._handle

    @classmethod
    def from_ipc_handle(cls, device: Any, handle: bytes) -> "StubEvent":
        ev = cls(interprocess=True)
        ev._handle = handle
        return ev

    def __repr__(self) -> str:
        return f"StubEvent(interprocess={self.interprocess}, recorded={self._recorded})"


class StubStream:
    def __init__(self, device: Any = "cpu", priority: int = 0, **kwargs: Any) -> None:
        self.device = device
        self.priority = priority
        self.cuda_stream = 0

    def synchronize(self) -> None:
        return None

    def wait_event(self, event: StubEvent) -> None:
        return None

    def wait_stream(self, stream: "StubStream") -> None:
        return None

    def record_event(self, event: StubEvent | None = None) -> StubEvent:
        event = event or StubEvent()
        event.record(self)
        return event

    def query(self) -> bool:
        return True

    @staticmethod
    def priority_range() -> tuple[int, int]:
        return (0, 0)

    def __repr__(self) -> str:
        return f"StubStream(device={self.device}, priority={self.priority})"


class StubCPUDevice:
    """Stub stand-in for torch_dev in CPU-only test environments."""

    def __init__(self, device_type: str = "cpu") -> None:
        self._device_type = device_type
        self._stream = StubStream(device=device_type)

        self.Event = StubEvent
        self.Stream = StubStream

    def is_available(self) -> bool:
        return False

    def init(self) -> None:
        """No-op matching torch_dev.init()."""
        return None

    def device(self, device: Any = None):
        """Context manager matching torch_dev.device()."""
        return nullcontext()

    def current_stream(self, device: Any = None) -> StubStream:
        return self._stream

    def default_stream(self, device: Any = None) -> StubStream:
        return self._stream

    def stream(self, stream: StubStream | None = None):
        return nullcontext(stream or self._stream)

    def synchronize(self, device: Any = None) -> None:
        return None

    def set_stream(self, stream: StubStream) -> None:
        self._stream = stream

    def device_count(self) -> int:
        return 1

    def current_device(self) -> int:
        return 0

    def set_device(self, device: Any) -> None:
        return None

    def get_device_properties(self, device: Any = 0) -> StubDeviceProperties:
        return StubDeviceProperties()

    def empty_cache(self) -> None:
        """No-op matching torch_dev.empty_cache()."""
        return None

    def __getattr__(self, name: str):
        raise AttributeError(f"StubCPUDevice does not implement '{name}'")

    def __repr__(self) -> str:
        return f"StubCPUDevice(device_type={self._device_type})"
