# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified ``DeviceOps`` abstraction and spec-based resolution.

These tests are the acceptance gate for the DeviceOps. They stay
platform-agnostic by exercising the torch baseline, the ``DeviceSpec``
dispatch, the ``lmcache.c_ops`` shim, and the ``_bind_native`` mechanism
without requiring any compiled accelerator module.
"""

# Standard
from typing import Any
import types

# Third Party
import pytest

# First Party
import lmcache.v1.platform as platform_pkg
from lmcache.v1.platform import resolve_device_ops_cls
from lmcache.v1.platform import _torch_ops
from lmcache.v1.platform.base_device_spec import DeviceSpec
from lmcache.v1.platform.base_device_ops import OPS, DeviceOps
from lmcache.v1.platform.cpu.device_ops import CpuDeviceOps
from lmcache.v1.platform.ops_types import (
    BatchStep,
    EngineKVFormat,
    KernelGroupSpec,
    LaunchVar,
    PageBufferShapeDesc,
    StagingCopy,
    TransferDirection,
)


@pytest.fixture
def isolated_registry() -> Any:
    """Snapshot the device-spec table so tests can install fakes safely."""
    saved = dict(platform_pkg._DEVICE_REGISTRY)
    try:
        yield saved
    finally:
        platform_pkg._DEVICE_REGISTRY.clear()
        platform_pkg._DEVICE_REGISTRY.update(saved)


@pytest.fixture
def target_module() -> types.ModuleType:
    """Fresh module object to use as populate_module target."""
    return types.ModuleType("test_target")


# -- Contract --------------------------------------------------------------


def test_ops_contract_has_36_names() -> None:
    assert len(OPS) == 36


def test_base_populates_every_op_and_type(target_module: types.ModuleType) -> None:
    """populate_module installs all ops as direct _torch_ops refs, plus types."""
    DeviceOps.populate_module(target_module)
    for name in OPS:
        fn = getattr(target_module, name)
        assert callable(fn), name
        # Must be the torch baseline function directly
        assert fn is getattr(_torch_ops, name), name
    assert target_module.TransferDirection is TransferDirection
    assert target_module.EngineKVFormat is EngineKVFormat
    assert target_module.GPUKVFormat is EngineKVFormat  # back-compat alias
    assert target_module.PageBufferShapeDesc is PageBufferShapeDesc
    assert target_module.StagingCopy is StagingCopy
    assert target_module.LaunchVar is LaunchVar
    assert target_module.BatchStep is BatchStep
    assert target_module.KernelGroupSpec is KernelGroupSpec
    assert callable(target_module.set_shape_desc_dtype)


def test_every_registered_device_populates_all_ops(
    isolated_registry: Any, target_module: types.ModuleType
) -> None:
    """Each discovered DeviceSpec resolves an ops class that populates all ops."""
    for device_type, spec in isolated_registry.items():
        m = types.ModuleType(f"test_{device_type}")
        spec.ops_cls.populate_module(m)
        for name in OPS:
            assert callable(getattr(m, name)), (device_type, name)


# -- Dispatch --------------------------------------------------------------


def test_cpu_uses_torch_baseline(target_module: types.ModuleType) -> None:
    """CpuDeviceOps adds no overrides: every op is the _torch_ops function."""
    CpuDeviceOps.populate_module(target_module)
    for name in OPS:
        fn = getattr(target_module, name)
        assert fn is getattr(_torch_ops, name), name


def test_musa_overrides_only_one_op(target_module: types.ModuleType) -> None:
    """MusaDeviceOps overrides exactly one hot op; the rest are baseline."""
    musa_mod = pytest.importorskip(
        "lmcache.v1.platform.musa.device_ops",
        reason="musa platform package unavailable",
    )
    musa_mod.MusaDeviceOps.populate_module(target_module)
    overridden = [
        name
        for name in OPS
        if getattr(target_module, name) is not getattr(_torch_ops, name)
    ]
    assert overridden == ["multi_layer_block_kv_transfer"]


def test_musa_shim_dispatches_native_first(
    target_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling multi_layer_block_kv_transfer through a populated module
    (the production c_ops shim path) dispatches via the native MUSA path
    when inputs are tensor-backed."""
    # Third Party
    import torch

    musa_mod = pytest.importorskip(
        "lmcache.v1.platform.musa.device_ops",
        reason="musa platform package unavailable",
    )
    native_mod = pytest.importorskip(
        "lmcache.v1.platform.musa.native_kv_transfer",
        reason="musa native_kv_transfer unavailable",
    )
    musa_mod.MusaDeviceOps.populate_module(target_module)

    captured: dict[str, Any] = {}

    def _fake_native(**kwargs: Any) -> bool:
        captured.update(kwargs)
        return True  # signal success -> skip fallback

    monkeypatch.setattr(
        native_mod,
        "try_native_multi_layer_block_kv_transfer",
        _fake_native,
    )

    paged = [torch.zeros(4, 4, 16) for _ in range(2)]
    objects = [torch.zeros(2, 8, 16)]

    shape_desc = PageBufferShapeDesc()
    shape_desc.nl = 2
    shape_desc.nb = 1
    shape_desc.bs = 4
    shape_desc.nh = 4
    shape_desc.hs = 16
    shape_desc.kv_size = 2
    shape_desc.element_size = torch.empty((), dtype=torch.float32).element_size()

    target_module.multi_layer_block_kv_transfer(
        paged,
        objects,
        [0, 1],
        "musa",
        TransferDirection.D2H,
        shape_desc,
        8,
        EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )

    # Native was called with tensor-backed inputs
    assert captured, "native path was not invoked through populated module"
    assert captured["direction"] == TransferDirection.D2H


# -- _bind_native ----------------------------------------------------------


class _FakeNativeModule:
    """Stand-in compiled module: a couple of real ops + a non-OPS symbol."""

    @staticmethod
    def multi_layer_kv_transfer(*a: Any, **k: Any) -> str:
        return "native-mlt"

    @staticmethod
    def calculate_cdf(*a: Any, **k: Any) -> str:
        return "native-cdf"

    @staticmethod
    def not_in_ops(*a: Any, **k: Any) -> str:
        return "ignored"


def test_bind_native_shadows_baseline_for_present_ops(
    target_module: types.ModuleType,
) -> None:
    DeviceOps.populate_module(target_module)
    DeviceOps._bind_native(target_module, _FakeNativeModule())
    assert target_module.multi_layer_kv_transfer() == "native-mlt"
    assert target_module.calculate_cdf() == "native-cdf"
    # Ops absent from native keep the torch baseline
    assert target_module.single_layer_kv_transfer is _torch_ops.single_layer_kv_transfer


def test_bind_native_ignores_symbols_absent_from_ops(
    target_module: types.ModuleType,
) -> None:
    DeviceOps.populate_module(target_module)
    DeviceOps._bind_native(target_module, _FakeNativeModule())
    assert not hasattr(target_module, "not_in_ops")


# -- DeviceSpec resolution -------------------------------------------------


def test_cpu_and_empty_string_resolve_to_expected_ops_classes() -> None:
    """``cpu`` resolves via CpuDeviceSpec; ``""`` uses the bare fallback."""
    assert resolve_device_ops_cls("cpu") is CpuDeviceOps
    assert resolve_device_ops_cls("") is DeviceOps


def test_cpu_without_registered_spec_falls_back_to_base_device_ops(
    isolated_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = {k: v for k, v in isolated_registry.items() if k != "cpu"}
    monkeypatch.setattr(platform_pkg, "_DEVICE_REGISTRY", table)
    assert resolve_device_ops_cls("cpu") is DeviceOps


def test_unregistered_accelerator_fails_fast(
    isolated_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A requested accelerator with no registered class is a hard error -- no
    silent degradation to the torch baseline."""
    table = {k: v for k, v in isolated_registry.items() if k != "cuda"}
    monkeypatch.setattr(platform_pkg, "_DEVICE_REGISTRY", table)
    with pytest.raises(
        RuntimeError,
        match="refusing to silently fall back to the torch baseline",
    ):
        resolve_device_ops_cls("cuda")


def test_unknown_accelerator_fails_fast_without_registry_edits() -> None:
    with pytest.raises(
        RuntimeError,
        match="refusing to silently fall back to the torch baseline",
    ):
        resolve_device_ops_cls("definitely-not-a-real-device")


def test_new_device_needs_zero_resolver_edits(
    isolated_registry: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scalability: a fresh DeviceSpec resolves with no resolver change."""

    class DummyDeviceOps(DeviceOps):
        device_type = "dummy"

    class DummyDeviceSpec(DeviceSpec):
        @property
        def device_type(self) -> str:
            return "dummy"

        @property
        def ops_cls(self) -> "type[DeviceOps]":
            return DummyDeviceOps

    monkeypatch.setattr(
        platform_pkg,
        "_DEVICE_REGISTRY",
        {**isolated_registry, "dummy": DummyDeviceSpec()},
    )
    assert resolve_device_ops_cls("dummy") is DummyDeviceOps


def test_empty_device_type_is_skipped_during_discovery() -> None:
    """The fallback/base spec is never registered as a concrete device."""
    assert "" not in platform_pkg._DEVICE_REGISTRY
    assert all(spec.device_type for spec in platform_pkg._DEVICE_REGISTRY.values())


# -- c_ops shim ------------------------------------------------------------


def test_c_ops_shim_exposes_full_surface() -> None:
    """``import lmcache.c_ops`` exposes all ops + types."""
    # First Party
    import lmcache.c_ops as c_ops

    for name in OPS:
        assert callable(getattr(c_ops, name)), name
    assert c_ops.TransferDirection is not None
    assert c_ops.EngineKVFormat is not None
    assert c_ops.GPUKVFormat is c_ops.EngineKVFormat
    assert c_ops.PageBufferShapeDesc is not None
    assert c_ops.StagingCopy is not None
    assert c_ops.LaunchVar is not None
    assert c_ops.BatchStep is not None
    assert c_ops.KernelGroupSpec is not None
    assert callable(c_ops.set_shape_desc_dtype)
