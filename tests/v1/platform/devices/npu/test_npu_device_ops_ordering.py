# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for NPU native binding and stream-ordered recording."""

# Standard
from collections.abc import Sequence
from types import ModuleType
import importlib.abc
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.platform.devices.npu import device_ops as npu_device_ops_module
from lmcache.v1.platform.devices.npu.device_ops import NpuDeviceOps


def _install_fake_npu_ops(monkeypatch: pytest.MonkeyPatch, **symbols: object) -> None:
    """Bind ``NpuDeviceOps`` against a fake ``lmcache.npu_ops``.

    The ``lmcache_ascend`` plugin is marked absent: importing the real one
    applies its patches, which these binding tests are not about.
    """
    fake_npu_ops = ModuleType("lmcache.npu_ops")
    for name, value in symbols.items():
        setattr(fake_npu_ops, name, value)
    monkeypatch.setitem(sys.modules, "lmcache.npu_ops", fake_npu_ops)
    # ``import lmcache.npu_ops as x`` reads the parent attribute, which the
    # real extension sets when it is imported. Point it at the fake too.
    monkeypatch.setattr(sys.modules["lmcache"], "npu_ops", fake_npu_ops, raising=False)
    monkeypatch.setitem(sys.modules, "lmcache_ascend", None)


def test_record_completion_syncs_stream_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = NpuDeviceOps()
    ops.drain_recorded_completions()  # isolate the shared fallback buffer
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        npu_device_ops_module,
        "_synchronize_npu_stream_pointer",
        lambda ptr: calls.append(("sync", ptr)),
    )

    ops.record_completion_on_stream(0xABCD, "finish_write", b"payload")

    assert calls == [("sync", 0xABCD)]
    assert ops.drain_recorded_completions() == [("finish_write", b"payload")]


def test_record_event_syncs_stream_before_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ops = NpuDeviceOps()
    ops.drain_recorded_events()
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        npu_device_ops_module,
        "_synchronize_npu_stream_pointer",
        lambda ptr: calls.append(("sync", ptr)),
    )

    ops.record_event_on_stream(
        0xBEEF, "mp.store.start", "session", {"device": "npu"}, {"engine_id": 1}
    )

    assert calls == [("sync", 0xBEEF)]
    events = ops.drain_recorded_events()
    assert len(events) == 1
    event_type, session_id, _timestamp, string_metadata, int_metadata = events[0]
    assert event_type == "mp.store.start"
    assert session_id == "session"
    assert string_metadata == {"device": "npu"}
    assert int_metadata == {"engine_id": 1}


def test_sync_helper_raises_on_acl_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingRt:
        @staticmethod
        def synchronize_stream(ptr: int) -> int:
            return 507899

    fake_acl = ModuleType("acl")
    fake_acl.rt = _FailingRt  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "acl", fake_acl)
    with pytest.raises(RuntimeError, match="507899"):
        npu_device_ops_module._synchronize_npu_stream_pointer(7)


def test_ensure_native_binds_native_ops_and_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native symbols take the place of the torch-baseline members."""

    def _native_recorder(stream_ptr: int, kind: str, payload: bytes) -> None:
        pass

    class _NativeBatchStep:
        pass

    _install_fake_npu_ops(
        monkeypatch,
        record_completion_on_stream=_native_recorder,
        BatchStep=_NativeBatchStep,
    )

    ops = NpuDeviceOps()
    ops.ensure_native()
    assert ops.record_completion_on_stream is _native_recorder
    assert ops.BatchStep is _NativeBatchStep


def test_ensure_native_raises_without_npu_ops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing lmcache.npu_ops must fail fast, not soft-fall back.

    The torch baseline reconstructs pointer-mode staging buffers as CPU
    tensors, which would silently transfer in the wrong address space on
    NPU.
    """
    monkeypatch.setitem(sys.modules, "lmcache_ascend", None)
    monkeypatch.setitem(sys.modules, "lmcache.npu_ops", None)
    ops = NpuDeviceOps()
    with pytest.raises(ImportError, match="lmcache.npu_ops"):
        ops.ensure_native()


def test_ensure_native_raises_when_plugin_fails_to_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed plugin that fails to import is an error, not a no-op.

    Its hardware-specific patches would otherwise be missing without any
    indication.
    """

    class _BrokenPluginFinder(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: Sequence[str] | None,
            target: ModuleType | None = None,
        ) -> None:
            if fullname == "lmcache_ascend":
                raise ModuleNotFoundError("No module named 'dep'", name="dep")
            return None

    monkeypatch.delitem(sys.modules, "lmcache_ascend", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BrokenPluginFinder(), *sys.meta_path])
    ops = NpuDeviceOps()
    with pytest.raises(ModuleNotFoundError, match="No module named 'dep'"):
        ops.ensure_native()
