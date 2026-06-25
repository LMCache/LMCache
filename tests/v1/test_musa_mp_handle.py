# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from typing import cast
import importlib

# Third Party
import pytest
import torch


class _FakeMusaEvent:
    """Minimal TorchMUSA Event facade for capability-gate tests."""

    def __init__(self, interprocess: bool = False) -> None:
        self.interprocess = interprocess

    @classmethod
    def from_ipc_handle(cls, _device: object, _handle: bytes) -> "_FakeMusaEvent":
        """Reconstruct an event from an IPC handle."""
        return cls(interprocess=True)


class _FakeDevice:
    """Minimal device object exposing only ``type`` for factory routing tests."""

    type = "musa"


class _FakeMusaTensor:
    """Minimal tensor-like object exposing a MUSA device for routing tests."""

    device = _FakeDevice()


def _fake_musa_kv_caches() -> dict[str, torch.Tensor]:
    """Return a typed KV cache mapping backed by a minimal MUSA tensor facade."""
    return cast(dict[str, torch.Tensor], {"layer_0": _FakeMusaTensor()})


def test_musa_handle_transfer_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage4 MUSA handle transfer is opt-in and unavailable by default."""
    # First Party
    from lmcache.v1.platform.musa import ipc

    monkeypatch.delenv(ipc.ENV_MUSA_HANDLE_TRANSFER, raising=False)

    assert ipc.is_musa_handle_transfer_enabled() is False
    assert ipc.is_musa_handle_transfer_available() is False


def test_musa_handle_transfer_requires_torch_musa(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opt-in alone is not enough without a visible TorchMUSA runtime."""
    # First Party
    from lmcache.v1.platform.musa import ipc

    monkeypatch.setenv(ipc.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc, "is_torch_musa_available", lambda: False)

    assert ipc.is_musa_handle_transfer_available() is False


def test_get_torch_musa_module_imports_torch_musa_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TorchMUSA may register ``torch.musa`` only after ``torch_musa`` import."""
    # First Party
    from lmcache.v1.platform.musa import ipc

    fake_torch = SimpleNamespace()
    torch_musa = SimpleNamespace(is_available=lambda: True)

    def import_module(name: str) -> object:
        assert name == "torch_musa"
        fake_torch.musa = torch_musa
        return SimpleNamespace()

    monkeypatch.setattr(ipc, "torch", fake_torch)
    monkeypatch.setattr(ipc.importlib, "import_module", import_module)

    assert ipc.get_torch_musa_module() is torch_musa
    assert ipc.is_torch_musa_available() is True


def test_musa_handle_transfer_requires_event_and_block_transfer_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Memory IPC alone is not enough to enable the MUSA handle path."""
    # First Party
    from lmcache.v1.platform.musa import ipc

    torch_musa = SimpleNamespace(
        ipc_get_mem_handle=lambda _tensor: b"handle",
        ipc_open_mem_handle=lambda _handle, nbytes, _device: torch.empty(
            nbytes,
            dtype=torch.uint8,
        ),
    )

    monkeypatch.setenv(ipc.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc, "is_torch_musa_available", lambda: True)
    monkeypatch.setattr(ipc, "get_torch_musa_module", lambda: torch_musa)

    assert ipc.check_torch_musa_ipc_support(torch_musa) is True
    assert ipc.check_torch_musa_event_support(torch_musa) is False
    assert ipc.is_musa_handle_transfer_available() is False


def test_musa_handle_transfer_available_when_all_capabilities_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced MUSA handle mode is available only after all gates pass."""
    # First Party
    from lmcache.v1.platform.musa import ipc

    torch_musa = SimpleNamespace(
        ipc_get_mem_handle=lambda _tensor: b"handle",
        ipc_open_mem_handle=lambda _handle, nbytes, _device: torch.empty(
            nbytes,
            dtype=torch.uint8,
        ),
        Event=_FakeMusaEvent,
    )

    monkeypatch.setenv(ipc.ENV_MUSA_HANDLE_TRANSFER, "1")
    monkeypatch.setattr(ipc, "is_torch_musa_available", lambda: True)
    monkeypatch.setattr(ipc, "get_torch_musa_module", lambda: torch_musa)
    monkeypatch.setattr(ipc, "is_musa_block_transfer_available", lambda: True)

    assert ipc.check_torch_musa_ipc_support(torch_musa) is True
    assert ipc.check_torch_musa_event_support(torch_musa) is True
    assert ipc.is_musa_handle_transfer_available() is True


def test_musa_handle_transfer_rejects_incomplete_torch_musa_runtime() -> None:
    """Missing TorchMUSA IPC symbols keep Stage4 handle mode disabled."""
    # First Party
    from lmcache.v1.platform.musa import ipc

    torch_musa = SimpleNamespace(ipc_get_mem_handle=lambda _tensor: b"handle")

    assert ipc.check_torch_musa_ipc_support(torch_musa) is False


def test_musa_platform_discovers_factory_and_registers_capability_predicate() -> None:
    """The MUSA wrapper is auto-discovered while availability stays explicit."""
    # First Party
    from lmcache.v1.platform import _registry as platform_registry
    import lmcache.v1.platform.musa as musa_platform

    snapshot = platform_registry.snapshot()
    try:
        platform_registry.reset_for_tests()
        importlib.reload(musa_platform)

        factory = platform_registry.get_kv_wrapper_factory("musa")

        assert platform_registry.is_available("musa") is False
        assert callable(factory)
        with pytest.raises(ValueError, match="expected a MUSA tensor"):
            factory(torch.empty(1))
    finally:
        platform_registry.restore(snapshot)


def test_create_transfer_context_auto_keeps_musa_on_data_path() -> None:
    """Stage4 must not silently switch MUSA auto mode away from Stage3 data path."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        EngineDrivenTransferContext,
        create_transfer_context,
    )

    context = create_transfer_context(_fake_musa_kv_caches())

    assert isinstance(context, EngineDrivenTransferContext)


def test_create_transfer_context_musa_handle_requires_capability() -> None:
    """Forced MUSA handle mode fails closed when the Stage4 capability is absent."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import create_transfer_context

    with pytest.raises(ValueError, match="not available"):
        create_transfer_context(_fake_musa_kv_caches(), mode="lmcache_driven")


def test_create_transfer_context_musa_handle_allowed_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced MUSA handle mode is allowed once the platform reports capability."""
    # First Party
    from lmcache.v1.multiprocess.transfer_context import (
        LMCacheDrivenTransferContext,
        create_transfer_context,
    )
    from lmcache.v1.platform import _registry as platform_registry

    snapshot = platform_registry.snapshot()
    try:
        platform_registry.register_kv_wrapper("musa", lambda tensor: tensor)
        platform_registry.register_availability("musa", lambda: True)

        context = create_transfer_context(
            _fake_musa_kv_caches(),
            mode="lmcache_driven",
        )
    finally:
        platform_registry.restore(snapshot)

    assert isinstance(context, LMCacheDrivenTransferContext)


def test_musa_ipc_wrapper_rejects_non_musa_tensor() -> None:
    """The MUSA IPC wrapper never accepts CPU tensors by accident."""
    # First Party
    from lmcache.v1.platform.musa.ipc import MusaIPCWrapper

    with pytest.raises(ValueError, match="expected a MUSA tensor"):
        MusaIPCWrapper(torch.empty(4))


def test_musa_ipc_wrapper_uses_device_agnostic_wire_base() -> None:
    """Stage4 uses the device-agnostic KVCache wire base for MUSA handles."""
    # First Party
    from lmcache.v1.multiprocess.custom_types import DeviceIPCWrapper, KVCache
    from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper
    from lmcache.v1.platform.musa.ipc import MusaIPCWrapper

    assert issubclass(MusaIPCWrapper, DeviceIPCWrapper)
    assert not issubclass(MusaIPCWrapper, CudaIPCWrapper)
    assert MusaIPCWrapper.device_type == "musa"
    assert KVCache == list[DeviceIPCWrapper]
