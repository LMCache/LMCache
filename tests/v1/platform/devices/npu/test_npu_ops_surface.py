# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the curated NPU ops binding surface."""

# Standard
import sys
import types

# Third Party
import pytest

# First Party
from lmcache.v1.platform import torch_ops

pytestmark = pytest.mark.no_shared_allocator

_NATIVE_ONLY_NAMES = (
    "TransferDirection",
    "EngineKVFormat",
    "GPUKVFormat",
    "is_cross_layer",
    "is_kv_list",
    "is_layer_list",
    "is_mla",
)


def _install_fake_npu_ops(monkeypatch: pytest.MonkeyPatch, **symbols: object) -> None:
    """Install a fake ``lmcache.npu_ops`` and force a fresh facade import."""
    fake = types.ModuleType("lmcache.npu_ops")
    for name, value in symbols.items():
        setattr(fake, name, value)
    monkeypatch.setitem(sys.modules, "lmcache.npu_ops", fake)
    # ``import lmcache.npu_ops as x`` reads the parent attribute, which the
    # real extension sets when it is imported. Point it at the fake too.
    monkeypatch.setattr(sys.modules["lmcache"], "npu_ops", fake, raising=False)
    monkeypatch.delitem(
        sys.modules, "lmcache.v1.platform.devices.npu.ops", raising=False
    )
    monkeypatch.delattr(
        sys.modules["lmcache.v1.platform.devices.npu"],
        "ops",
        raising=False,
    )


def _load_surface() -> types.ModuleType:
    """Import the facade against the currently installed fake extension."""
    # First Party
    import lmcache.v1.platform.devices.npu.ops as ops_surface

    return ops_surface


def test_surface_excludes_native_only_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CppDesc:
        pass

    def _native_transfer(*args: object) -> None:
        pass

    symbols: dict[str, object] = {
        "PageBufferShapeDesc": _CppDesc,
        "multi_layer_block_kv_transfer": _native_transfer,
    }
    symbols.update({name: object() for name in _NATIVE_ONLY_NAMES})
    _install_fake_npu_ops(monkeypatch, **symbols)

    surface = _load_surface()
    for name in _NATIVE_ONLY_NAMES:
        assert not hasattr(surface, name)
    assert surface.PageBufferShapeDesc is _CppDesc
    assert surface.multi_layer_block_kv_transfer is _native_transfer


def test_torch_fallback_reexports_are_filtered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CppDesc:
        pass

    _install_fake_npu_ops(
        monkeypatch,
        PageBufferShapeDesc=_CppDesc,
        record_completion_on_stream=torch_ops.record_completion_on_stream,
    )

    surface = _load_surface()
    assert not hasattr(surface, "record_completion_on_stream")
    assert surface.PageBufferShapeDesc is _CppDesc


def test_genuine_native_symbols_are_kept(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _native_recorder(stream_ptr: int, kind: str, payload: bytes) -> None:
        pass

    _install_fake_npu_ops(
        monkeypatch,
        record_completion_on_stream=_native_recorder,
    )

    surface = _load_surface()
    assert surface.record_completion_on_stream is _native_recorder
