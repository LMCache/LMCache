# SPDX-License-Identifier: Apache-2.0
"""CPU-only and optional runtime tests for MUSA host-memory pinning."""

# Standard
from collections.abc import Callable
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed import config as distributed_config
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.memory_allocators import lazy_memory_allocator
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.musa import MusaDeviceSpec
from lmcache.v1.platform.musa import pin_memory as pin_memory_module
from lmcache.v1.platform.musa.pin_memory import MusaPinMemoryBackend


class _FakeMusart:
    """Small TorchMUSA runtime double that records host-registration calls."""

    def __init__(
        self,
        register_result: int | Exception = 0,
        unregister_result: int | Exception = 0,
    ) -> None:
        self.register_result = register_result
        self.unregister_result = unregister_result
        self.register_calls: list[tuple[int, int, int]] = []
        self.unregister_calls: list[int] = []

    def musaHostRegister(self, ptr: int, size: int, flags: int) -> int:
        """Record and emulate ``musaHostRegister``."""
        self.register_calls.append((ptr, size, flags))
        if isinstance(self.register_result, Exception):
            raise self.register_result
        return self.register_result

    def musaHostUnregister(self, ptr: int) -> int:
        """Record and emulate ``musaHostUnregister``."""
        self.unregister_calls.append(ptr)
        if isinstance(self.unregister_result, Exception):
            raise self.unregister_result
        return self.unregister_result


def _install_musart(
    monkeypatch: pytest.MonkeyPatch,
    musart: object,
) -> None:
    """Expose a fake ``torch.musa.musart()`` binding for one test."""
    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(musart=lambda: musart),
        raising=False,
    )


def _capture_warnings(
    monkeypatch: pytest.MonkeyPatch,
    logger: object,
) -> list[str]:
    """Replace one LMCache logger warning method with a message recorder."""
    messages: list[str] = []

    def _record(message: str, *args: object) -> None:
        messages.append(message % args)

    monkeypatch.setattr(logger, "warning", _record)
    return messages


def test_is_pin_supported_when_both_runtime_functions_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both host-registration entry points enable the MUSA capability."""
    _install_musart(monkeypatch, _FakeMusart())

    assert MusaPinMemoryBackend().is_pin_supported is True


def test_is_not_pin_supported_when_musart_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing musart lookup is a safe unsupported result."""

    def _raise() -> object:
        raise RuntimeError("musart unavailable")

    monkeypatch.setattr(
        torch,
        "musa",
        SimpleNamespace(musart=_raise),
        raising=False,
    )

    assert MusaPinMemoryBackend().is_pin_supported is False


def test_is_not_pin_supported_when_register_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime without ``musaHostRegister`` is unsupported."""
    _install_musart(
        monkeypatch,
        SimpleNamespace(musaHostUnregister=lambda ptr: 0),
    )

    assert MusaPinMemoryBackend().is_pin_supported is False


def test_is_not_pin_supported_when_unregister_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime without ``musaHostUnregister`` is unsupported."""
    _install_musart(
        monkeypatch,
        SimpleNamespace(musaHostRegister=lambda ptr, size, flags: 0),
    )

    assert MusaPinMemoryBackend().is_pin_supported is False


def test_pin_memory_forwards_pointer_size_and_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host registration forwards its complete runtime contract unchanged."""
    musart = _FakeMusart()
    _install_musart(monkeypatch, musart)

    assert MusaPinMemoryBackend().pin_memory(0x1000, 64 << 20, 0x02) is True
    assert musart.register_calls == [(0x1000, 64 << 20, 0x02)]


def test_pin_memory_returns_false_for_nonzero_runtime_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero MUSA registration status is not reported as success."""
    _install_musart(monkeypatch, _FakeMusart(register_result=1))

    assert MusaPinMemoryBackend().pin_memory(0x1000, 4096, 0x02) is False


def test_pin_memory_returns_false_when_runtime_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registration exception is logged and safely mapped to ``False``."""
    _install_musart(
        monkeypatch,
        _FakeMusart(register_result=RuntimeError("register failed")),
    )
    warnings = _capture_warnings(monkeypatch, pin_memory_module.logger)

    result = MusaPinMemoryBackend().pin_memory(0x1234, 4096, 0x02)

    assert result is False
    assert "musaHostRegister failed" in warnings[0]
    assert "ptr=0x1234" in warnings[0]


def test_unpin_memory_forwards_base_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host unregistration forwards the registered base pointer unchanged."""
    musart = _FakeMusart()
    _install_musart(monkeypatch, musart)

    assert MusaPinMemoryBackend().unpin_memory(0x2000) is True
    assert musart.unregister_calls == [0x2000]


def test_unpin_memory_returns_false_for_nonzero_runtime_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonzero unregistration status fails visibly."""
    _install_musart(monkeypatch, _FakeMusart(unregister_result=713))
    warnings = _capture_warnings(monkeypatch, pin_memory_module.logger)

    result = MusaPinMemoryBackend().unpin_memory(0x2000)

    assert result is False
    assert "musaHostUnregister returned error code 713" in warnings[0]


def test_unpin_memory_returns_false_when_runtime_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unregistration exception is logged and safely mapped to ``False``."""
    _install_musart(
        monkeypatch,
        _FakeMusart(unregister_result=RuntimeError("unregister failed")),
    )
    warnings = _capture_warnings(monkeypatch, pin_memory_module.logger)

    result = MusaPinMemoryBackend().unpin_memory(0x2345)

    assert result is False
    assert "musaHostUnregister failed" in warnings[0]
    assert "ptr=0x2345" in warnings[0]


def test_operations_return_false_when_backend_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported backends never attempt or claim pinning operations."""
    monkeypatch.delattr(torch, "musa", raising=False)
    backend = MusaPinMemoryBackend()

    assert backend.pin_memory(0x1000, 4096, 0x02) is False
    assert backend.unpin_memory(0x1000) is False


def test_musa_device_spec_uses_musa_pin_memory_backend() -> None:
    """The MUSA device specification registers the MUSA pinning adapter."""
    assert MusaDeviceSpec().pin_memory_backend is MusaPinMemoryBackend


def test_musa_device_spec_reports_pin_support_from_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The device capability gate reflects the discovered TorchMUSA API."""
    _install_musart(monkeypatch, _FakeMusart())

    assert MusaDeviceSpec().is_pin_supported is True


def test_musa_device_spec_does_not_affect_other_platforms() -> None:
    """The fallback device specification remains safely unsupported."""
    spec = DeviceSpec()

    assert spec.pin_memory_backend is None
    assert spec.is_pin_supported is False


def test_lazy_config_remains_enabled_when_musa_backend_is_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete MUSA binding keeps requested lazy allocation enabled."""
    _install_musart(monkeypatch, _FakeMusart())
    monkeypatch.setattr(
        distributed_config,
        "current_device_spec",
        MusaDeviceSpec(),
    )

    config = L1MemoryManagerConfig(size_in_bytes=1 << 30, use_lazy=True)

    assert config.use_lazy is True


def test_lazy_config_is_disabled_when_musa_binding_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An incomplete MUSA binding preserves the existing safe fallback."""
    monkeypatch.delattr(torch, "musa", raising=False)
    monkeypatch.setattr(
        distributed_config,
        "current_device_spec",
        MusaDeviceSpec(),
    )
    warnings = _capture_warnings(monkeypatch, distributed_config.logger)

    config = L1MemoryManagerConfig(size_in_bytes=1 << 30, use_lazy=True)

    assert config.use_lazy is False
    assert "Disabling l1-use-lazy" in warnings[0]


def test_lazy_allocator_pins_and_unpins_only_successful_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy allocation uses mapped registration and joins before unpinning."""
    chunk_size = 4096
    events: list[str] = []

    class _FakeDeviceSpec:
        is_pin_supported = True

        def __init__(self) -> None:
            self.pin_results = iter((True, False, True))
            self.pin_calls: list[tuple[int, int, int]] = []
            self.unpin_calls: list[int] = []

        def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
            self.pin_calls.append((ptr, size, flags))
            return next(self.pin_results)

        def unpin_memory(self, ptr: int) -> bool:
            events.append("unpin")
            self.unpin_calls.append(ptr)
            return True

    class _InlineThread:
        def __init__(
            self,
            *,
            target: Callable[[], None],
            daemon: bool,
            name: str,
        ) -> None:
            self.target = target

        def start(self) -> None:
            self.target()

        def join(self) -> None:
            events.append("join")

    fake_spec = _FakeDeviceSpec()
    monkeypatch.setattr(lazy_memory_allocator, "current_device_spec", fake_spec)
    monkeypatch.setattr(lazy_memory_allocator.threading, "Thread", _InlineThread)
    monkeypatch.setattr(
        lazy_memory_allocator.LazyMemoryAllocator,
        "PIN_CHUNK_SIZE",
        chunk_size,
    )
    monkeypatch.setattr(
        lazy_memory_allocator.LazyMemoryAllocator,
        "COMMIT_SIZE",
        2 * chunk_size,
    )

    allocator = lazy_memory_allocator.LazyMemoryAllocator(
        init_size=chunk_size,
        final_size=3 * chunk_size,
        align_bytes=chunk_size,
    )
    events.clear()
    allocator.close()

    base_ptr = fake_spec.pin_calls[0][0]
    assert fake_spec.pin_calls == [
        (base_ptr, chunk_size, 0x02),
        (base_ptr + chunk_size, chunk_size, 0x02),
        (base_ptr + 2 * chunk_size, chunk_size, 0x02),
    ]
    assert fake_spec.unpin_calls == [base_ptr, base_ptr + 2 * chunk_size]
    assert events == ["join", "unpin", "unpin"]


@pytest.mark.musa
def test_real_musa_runtime_registers_and_unregisters_64_mib() -> None:
    """Optionally smoke-test one real 64 MiB MUSA registration lifecycle."""
    pytest.importorskip("torch_musa")
    backend = MusaPinMemoryBackend()
    if not backend.is_pin_supported:
        pytest.skip("TorchMUSA host-registration binding is unavailable")

    buffer = torch.empty(64 << 20, dtype=torch.uint8, device="cpu")
    ptr = buffer.data_ptr()
    assert backend.pin_memory(ptr, buffer.nbytes, 0x02) is True
    assert backend.unpin_memory(ptr) is True
