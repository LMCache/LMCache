# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the unified ``DeviceOps`` abstraction and its registry.

These tests are the acceptance gate for the DeviceOps. They stay
platform-agnostic by exercising the torch baseline, the registry dispatch, the
``lmcache.c_ops`` shim, and the ``_bind_native`` mechanism without requiring any
compiled accelerator module.
"""

# Standard
from typing import Any

# Third Party
import pytest

# First Party
from lmcache.v1.platform._device_ops_registry import (
    get_device_ops_cls,
    restore_device_ops,
    snapshot_device_ops,
)
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
    """Snapshot the device-ops table so each test can install fakes without
    polluting other tests / the production discovery result."""
    saved = snapshot_device_ops()
    try:
        yield saved
    finally:
        restore_device_ops(saved)


# -- Contract --------------------------------------------------------------


def test_base_exposes_every_op_and_type() -> None:
    """Every op in ``OPS`` is a real callable on the base, plus shared types."""
    base = DeviceOps()
    for name in OPS:
        assert callable(getattr(base, name)), name
    assert base.TransferDirection is TransferDirection
    assert base.EngineKVFormat is EngineKVFormat
    assert base.GPUKVFormat is EngineKVFormat  # back-compat alias
    assert base.PageBufferShapeDesc is PageBufferShapeDesc
    assert base.StagingCopy is StagingCopy
    assert base.LaunchVar is LaunchVar
    assert base.BatchStep is BatchStep
    assert base.KernelGroupSpec is KernelGroupSpec
    assert callable(base.set_shape_desc_dtype)


def test_every_registered_device_satisfies_the_contract(
    isolated_registry: Any,
) -> None:
    """Each discovered DeviceOps subclass resolves all ops to callables."""
    for device_type, cls in isolated_registry.items():
        instance = cls()
        for name in OPS:
            assert callable(getattr(instance, name)), (device_type, name)


# -- Dispatch --------------------------------------------------------------


def test_cpu_inherits_the_torch_baseline() -> None:
    """CpuDeviceOps adds no overrides: every op is the base method."""
    cpu = CpuDeviceOps()
    for name in OPS:
        bound = getattr(cpu, name)
        assert bound.__qualname__ == "DeviceOps.%s" % name, name


def test_musa_overrides_only_one_op() -> None:
    """MusaDeviceOps overrides exactly one hot op; the rest are baseline."""
    musa_mod = pytest.importorskip(
        "lmcache.v1.platform.musa.device_ops",
        reason="musa platform package unavailable",
    )
    musa = musa_mod.MusaDeviceOps()
    overridden = [
        name
        for name in OPS
        if getattr(musa, name).__qualname__ != "DeviceOps.%s" % name
    ]
    assert overridden == ["multi_layer_block_kv_transfer"]


# -- _bind_native ----------------------------------------------------------


class _FakeNativeModule:
    """Stand-in compiled module: a couple of real ops + a non-OPS symbol."""

    def multi_layer_kv_transfer(self, *a: Any, **k: Any) -> str:
        return "native-mlt"

    def calculate_cdf(self, *a: Any, **k: Any) -> str:
        return "native-cdf"

    def not_in_ops(self, *a: Any, **k: Any) -> str:
        return "ignored"


def test_bind_native_shadows_baseline_for_present_ops() -> None:
    ops = DeviceOps()
    ops._bind_native(_FakeNativeModule())
    assert ops.multi_layer_kv_transfer() == "native-mlt"
    assert ops.calculate_cdf() == "native-cdf"
    assert ops.single_layer_kv_transfer.__qualname__ == (
        "DeviceOps.single_layer_kv_transfer"
    )


def test_bind_native_ignores_symbols_absent_from_ops() -> None:
    ops = DeviceOps()
    ops._bind_native(_FakeNativeModule())
    assert "not_in_ops" not in vars(ops)


# -- Registry --------------------------------------------------------------


def test_cpu_and_empty_resolve_to_baseline(isolated_registry: Any) -> None:
    assert get_device_ops_cls("cpu") is CpuDeviceOps
    assert get_device_ops_cls("") is DeviceOps


def test_unregistered_accelerator_fails_fast(isolated_registry: Any) -> None:
    """A requested accelerator with no registered class is a hard error -- no
    silent degradation to the torch baseline."""
    table = {k: v for k, v in isolated_registry.items() if k != "cuda"}
    restore_device_ops(table)
    with pytest.raises(RuntimeError, match="No DeviceOps class registered"):
        get_device_ops_cls("cuda")


def test_new_device_needs_zero_resolver_edits(isolated_registry: Any) -> None:
    """Scalability: a fresh DeviceOps subclass resolves through the same
    data-driven table with no resolver change (mirrors vLLM's dummy platform)."""

    class DummyDeviceOps(DeviceOps):
        device_type = "dummy"

    restore_device_ops({**isolated_registry, "dummy": DummyDeviceOps})
    assert get_device_ops_cls("dummy") is DummyDeviceOps


def test_empty_device_type_is_skipped_during_discovery() -> None:
    """The base (empty ``device_type``) is never registered as a device."""
    table = snapshot_device_ops()
    assert "" not in table
    assert DeviceOps not in table.values()


# -- c_ops shim ------------------------------------------------------------


def test_c_ops_shim_exposes_full_surface() -> None:
    """``import lmcache.c_ops`` exposes all ops + 3 types + GPUKVFormat."""
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
