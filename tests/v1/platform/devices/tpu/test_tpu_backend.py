# SPDX-License-Identifier: Apache-2.0
"""Tests for the Google TPU (TorchTPU) platform backend.

The spec tests run on any host (TorchTPU is not required). The ``tpu`` marked
round trip needs TorchTPU and a TPU, and skips otherwise.
"""

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_mode import MPTransferMode
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform._device_detect import (
    DEVICE_BACKEND_ENV_VAR,
    get_device_spec,
    get_torch_device,
)
from lmcache.v1.platform.devices.tpu import TpuDeviceSpec
from lmcache.v1.platform.devices.tpu.device_ops import TpuDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

TransferDirection = lmcache_native.TransferDirection

NUM_LAYERS = 2
NUM_BLOCKS = 16
BLOCK_SIZE = 4
NHP, P, HS = 2, 2, 128
DTYPE = torch.bfloat16
# Out of order and non-consecutive, as vLLM hands them out.
SCATTERED_BLOCKS = [13, 2, 9, 4, 0, 15, 6, 11, 3]
# ``torch.tpu`` functions strict TorchTPU rejects from callers outside torch.
INTERNAL_DEVICE_APIS = (
    "is_available",
    "device_count",
    "synchronize",
    "current_device",
    "set_device",
    "current_stream",
)


def _internal_api(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("is internal to TorchTPU")


def _strict_torch_tpu(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install an importable ``torch_tpu``, a ``torch.tpu`` whose functions
    raise like strict TorchTPU does for callers outside torch, and a
    ``torch.accelerator`` whose current accelerator is a TPU.

    Returns:
        MagicMock: The fake ``torch.accelerator``.
    """
    accelerator = MagicMock(name="torch.accelerator")
    accelerator.current_accelerator.return_value = SimpleNamespace(type="tpu")
    tpu = SimpleNamespace(**{name: _internal_api for name in INTERNAL_DEVICE_APIS})
    monkeypatch.setitem(sys.modules, "torch_tpu", ModuleType("torch_tpu"))
    monkeypatch.setattr(torch, "tpu", tpu, raising=False)
    monkeypatch.setattr(torch, "accelerator", accelerator)
    return accelerator


def _is_available_with(
    monkeypatch: pytest.MonkeyPatch, accelerator: object, torch_tpu: object
) -> bool:
    """Run ``is_available`` with ``torch_tpu`` stubbed (``None``: not installed)
    and ``torch.accelerator`` reporting ``accelerator``.

    ``torch.tpu`` raises like strict TorchTPU does for callers outside the
    framework, so a spec that consulted it would fail.
    """
    _strict_torch_tpu(monkeypatch).current_accelerator.return_value = accelerator
    monkeypatch.setitem(sys.modules, "torch_tpu", torch_tpu)
    return TpuDeviceSpec().is_available()


def _shape_desc() -> PageBufferShapeDesc:
    desc = PageBufferShapeDesc()
    desc.kv_size = 1
    desc.nl = NUM_LAYERS
    desc.nb = NUM_BLOCKS
    desc.bs = BLOCK_SIZE
    desc.nh = NHP * P // 2
    desc.hs = 2 * HS
    desc.element_size = DTYPE.itemsize
    desc.dtype = DTYPE
    return desc


# -- Spec contract ---------------------------------------------------------


def test_tpu_spec_properties() -> None:
    spec = TpuDeviceSpec()
    assert spec.device_type == "tpu"
    assert spec.torch_module_name == "tpu"
    assert spec.ops_cls is TpuDeviceOps


def test_spec_is_discovered_by_the_registry() -> None:
    """Defining the subclass is enough -- no manual registration needed."""
    assert isinstance(get_device_spec("tpu"), TpuDeviceSpec)
    assert type(resolve_device_ops("tpu")) is TpuDeviceOps


# -- Availability ----------------------------------------------------------


def test_unavailable_without_torch_tpu(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _is_available_with(monkeypatch, SimpleNamespace(type="tpu"), None) is False


def test_unavailable_when_the_accelerator_is_not_a_tpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_tpu = ModuleType("torch_tpu")
    assert _is_available_with(monkeypatch, None, torch_tpu) is False
    cuda = SimpleNamespace(type="cuda")
    assert _is_available_with(monkeypatch, cuda, torch_tpu) is False


def test_available_without_the_internal_torch_tpu_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tpu = SimpleNamespace(type="tpu")
    assert _is_available_with(monkeypatch, tpu, ModuleType("torch_tpu")) is True


# -- torch.tpu shims -------------------------------------------------------


def test_detected_device_module_reaches_torch_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LMCache's generic ``torch_dev`` calls work against strict TorchTPU."""
    accelerator = _strict_torch_tpu(monkeypatch)
    accelerator.device_count.return_value = 4
    monkeypatch.setenv(DEVICE_BACKEND_ENV_VAR, "tpu")
    get_torch_device.cache_clear()
    try:
        torch_dev, device_type = get_torch_device()
    finally:
        get_torch_device.cache_clear()

    assert device_type == "tpu"
    torch_dev.synchronize(device=1)
    accelerator.synchronize.assert_called_once_with(1)
    assert torch_dev.device_count() == 4
    assert torch_dev.current_stream() is accelerator.current_stream.return_value
    accelerator.current_stream.assert_called_once_with(None)


def test_installing_the_shims_twice_is_harmless(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _strict_torch_tpu(monkeypatch)
    spec = TpuDeviceSpec()
    assert spec.is_available() is True
    assert spec.is_available() is True

    torch.tpu.is_available()
    torch.tpu.current_device()
    torch.tpu.set_device(1)
    torch.tpu.synchronize()

    accelerator.is_available.assert_called_once_with()
    accelerator.current_device_index.assert_called_once_with()
    accelerator.set_device_index.assert_called_once_with(1)
    accelerator.synchronize.assert_called_once_with(None)


def test_shims_are_not_installed_without_a_tpu_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    accelerator = _strict_torch_tpu(monkeypatch)
    accelerator.current_accelerator.return_value = SimpleNamespace(type="cuda")

    assert TpuDeviceSpec().is_available() is False
    with pytest.raises(RuntimeError, match="internal to TorchTPU"):
        torch.tpu.synchronize()


# -- Engine-driven only ----------------------------------------------------


def test_tpu_is_engine_driven_only() -> None:
    spec = TpuDeviceSpec()
    assert spec.is_lmcache_driven_available() is False
    assert spec.default_mp_transfer_mode() is MPTransferMode.ENGINE_DRIVEN
    assert spec.ipc_wrapper_cls is None
    assert spec.event_ipc_backend is None


# -- Hardware (skipped without a TPU) --------------------------------------


@pytest.mark.tpu
def test_round_trip_on_device_compiles_once_per_bucket() -> None:
    """Round trip through TPU memory. Block counts that pad to the same power
    of two reuse the programs compiled for the first; a larger count compiles
    new ones."""
    if not TpuDeviceSpec().is_available():
        pytest.skip("no TorchTPU device")
    generator = torch.Generator().manual_seed(3)
    shape = (NUM_BLOCKS, BLOCK_SIZE, NHP, P, HS * DTYPE.itemsize)
    host = [
        torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator).view(DTYPE)
        for _ in range(NUM_LAYERS)
    ]
    device = [t.to("tpu") for t in host]

    def round_trip(num_blocks: int) -> None:
        blocks = SCATTERED_BLOCKS[:num_blocks]
        chunk_tokens = num_blocks * BLOCK_SIZE
        chunk = torch.zeros((NUM_LAYERS, chunk_tokens, NHP * P * HS), dtype=DTYPE)
        cleared = [torch.zeros_like(t) for t in device]
        for layers, direction in (
            (device, TransferDirection.D2H),
            (cleared, TransferDirection.H2D),
        ):
            TpuDeviceOps().multi_layer_block_kv_transfer(
                layers,
                [chunk],
                blocks,
                layers[0].device,
                direction,
                _shape_desc(),
                chunk_tokens,
                lmcache_native.EngineKVFormat.NL_X_NB_BS_NHP_P_HS,
                0,
            )
        torch.accelerator.synchronize()
        for got, expected in zip(cleared, host, strict=True):
            got_host = got.cpu()
            assert torch.equal(
                got_host[blocks].view(torch.uint8), expected[blocks].view(torch.uint8)
            )

    # TorchTPU exposes its compile-cache counters only as private module
    # functions; its own tests count compiles the same way.
    round_trip(5)
    misses = torch.tpu._get_cache_misses()
    for num_blocks in (6, 7, 8):
        round_trip(num_blocks)
    assert torch.tpu._get_cache_misses() == misses
    round_trip(9)
    assert torch.tpu._get_cache_misses() > misses
