# SPDX-License-Identifier: Apache-2.0
"""CPU-runnable capability and ordering contracts for the MLU backend."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock
import importlib

# Third Party
import pytest
import torch

# First Party
from lmcache.lmcache_native import EngineKVFormat, TransferDirection
from lmcache.v1.multiprocess.transfer_mode import MPTransferMode
from lmcache.v1.platform import get_device_spec, resolve_device_ops
from lmcache.v1.platform.devices.mlu import MluDeviceSpec
from lmcache.v1.platform.devices.mlu.device_ops import MluDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc

pytestmark = pytest.mark.no_shared_allocator


def test_discovery_and_capabilities() -> None:
    """Discovery finds MLU without advertising an unimplemented IPC stack."""
    spec = get_device_spec("mlu")
    assert isinstance(spec, MluDeviceSpec)
    assert spec.backend_name == "mlu"
    assert spec.torch_module_name == "mlu"
    assert isinstance(resolve_device_ops("mlu"), MluDeviceOps)
    assert spec.default_mp_transfer_mode() == MPTransferMode.ENGINE_DRIVEN
    assert not spec.is_lmcache_driven_available()
    assert spec.ipc_wrapper_cls is None
    assert spec.event_ipc_backend is None
    assert not spec.is_pin_supported


@pytest.mark.parametrize("available", [False, True])
def test_probe_registered_runtime(
    monkeypatch: pytest.MonkeyPatch, available: bool
) -> None:
    """A registered torch_mlu runtime owns the hardware availability check."""
    monkeypatch.setattr(
        torch, "mlu", SimpleNamespace(is_available=lambda: available), raising=False
    )
    assert MluDeviceSpec().is_available() is available


def test_probe_imports_optional_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """The availability probe can register torch_mlu on older torch versions."""
    monkeypatch.delattr(torch, "mlu", raising=False)
    real_import = importlib.import_module

    def register(name: str, package: str | None = None) -> object:
        if name == "torch_mlu":
            module = SimpleNamespace(is_available=lambda: True)
            monkeypatch.setattr(torch, "mlu", module, raising=False)
            return module
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", register)
    assert MluDeviceSpec().is_available()


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError])
def test_unavailable_runtime_is_not_selected(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    """Missing wheels, shared libraries, and drivers leave the backend unavailable."""
    monkeypatch.delattr(torch, "mlu", raising=False)
    monkeypatch.setattr(
        importlib, "import_module", Mock(side_effect=error("unavailable"))
    )
    assert not MluDeviceSpec().is_available()


def test_native_stream_handle() -> None:
    """The handle is the supplied MLU queue, including the default queue."""
    spec = MluDeviceSpec()
    for handle in (0, 123):
        assert spec.get_stream_handle(SimpleNamespace(mlu_stream=handle)) == handle
    with pytest.raises(TypeError, match="mlu_stream"):
        spec.get_stream_handle(SimpleNamespace(cuda_stream=123))


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("event", [False, True])
@pytest.mark.parametrize("handle", [0, 123])
def test_publication_waits_for_stream(
    monkeypatch: pytest.MonkeyPatch, fail: bool, event: bool, handle: int
) -> None:
    """Nothing is published before synchronization or after a synchronization error."""
    ops = MluDeviceOps()
    ops.drain_recorded_completions()
    ops.drain_recorded_events()
    observed: list[int] = []

    class Stream:
        def __init__(self, ptr: int) -> None:
            observed.append(ptr)

        def synchronize(self) -> None:
            assert ops.drain_recorded_completions() == []
            assert ops.drain_recorded_events() == []
            if fail:
                raise RuntimeError("queue failed")

    monkeypatch.setattr(
        torch,
        "mlu",
        SimpleNamespace(ExternalStream=Stream, default_stream=lambda: Stream(999)),
        raising=False,
    )

    def publish() -> None:
        if event:
            ops.record_event_on_stream(
                handle, "store", "session", {"key": "value"}, {"n": 1}
            )
        else:
            ops.record_completion_on_stream(handle, "store", b"payload")

    if fail:
        with pytest.raises(RuntimeError, match="queue failed"):
            publish()
        assert ops.drain_recorded_completions() == []
        assert ops.drain_recorded_events() == []
    else:
        publish()
        if event:
            records = ops.drain_recorded_events()
            assert len(records) == 1
            name, session, timestamp, strings, ints = records[0]
            assert (name, session, strings, ints) == (
                "store",
                "session",
                {"key": "value"},
                {"n": 1},
            )
            assert timestamp > 0
        else:
            assert ops.drain_recorded_completions() == [("store", b"payload")]
    assert observed == [999 if handle == 0 else handle]


@pytest.mark.parametrize("raw_paged", [False, True])
def test_block_transfer_rejects_addresses(raw_paged: bool) -> None:
    """Neither device KV nor staging addresses may enter the host-pointer fallback."""
    pages = torch.tensor([1234]) if raw_paged else [torch.zeros(2, 4, 2, 1, 4)]
    chunks = [torch.zeros(2, 1, 4, 4)] if raw_paged else [1234]
    with pytest.raises(TypeError, match="tensor"):
        MluDeviceOps().multi_layer_block_kv_transfer(
            pages,
            chunks,
            [0, 1],
            "cpu",
            TransferDirection.D2H,
            PageBufferShapeDesc(),
            4,
            EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            0,
        )


@pytest.mark.parametrize("direction", [TransferDirection.H2D, TransferDirection.D2H])
@pytest.mark.parametrize(
    "fmt",
    [
        EngineKVFormat.NL_X_NB_BSV_BSS,
        EngineKVFormat.NL_X_NBBS_ONE_HS,
        EngineKVFormat.NL_X_NP_X_NB_BS_ONE_HS,
        EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS,
        EngineKVFormat.NL_X_NB_NH_BS_CS,
        EngineKVFormat.NL_X_NB_BS_NH_CS,
        EngineKVFormat.NL_X_NB_NH_BS_TWO_HS,
        EngineKVFormat.NL_X_NB_BS_NH_TWO_HS,
        EngineKVFormat.NL_X_NB_TWO_BS_NH_HS,
        EngineKVFormat.NL_X_NB_TWO_NH_BS_HS,
        EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS,
        EngineKVFormat.NB_NL_TWO_BS_NH_HS,
        EngineKVFormat.NB_NL_TWO_NH_BS_HS,
        EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS,
        EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS,
        EngineKVFormat.NL_X_NB_BS_NH_HS,
    ],
    ids=lambda fmt: fmt.name,
)
def test_block_transfer_rejects_unsupported_layouts_before_copy(
    fmt: EngineKVFormat, direction: TransferDirection
) -> None:
    """Unsupported layout contracts cannot modify either tensor operand."""
    pages = [torch.ones(2, 4, 2, 1, 4)]
    chunks = [torch.full((2, 1, 4, 4), -1.0)]
    desc = PageBufferShapeDesc()
    desc.nl, desc.nb, desc.bs = 1, 4, 2
    desc.nh, desc.hs, desc.kv_size = 1, 4, 2
    desc.element_size, desc.dtype = 4, torch.float32
    with pytest.raises(NotImplementedError, match="Unsupported MLU KV format"):
        MluDeviceOps().multi_layer_block_kv_transfer(
            pages, chunks, [0, 1], "cpu", direction, desc, 4, fmt, 0
        )
    assert torch.equal(pages[0], torch.ones_like(pages[0]))
    assert torch.equal(chunks[0], torch.full_like(chunks[0], -1))


@pytest.mark.parametrize("direction", [TransferDirection.H2D, TransferDirection.D2H])
@pytest.mark.parametrize("singleton_layer", [0, 1])
def test_block_transfer_rejects_singleton_hnd_before_copy(
    singleton_layer: int, direction: TransferDirection
) -> None:
    """HND traits cannot admit an extra axis, even in a later layer."""
    pages = [torch.ones(2, 4, 3, 2, 8) for _ in range(2)]
    pages[singleton_layer] = pages[singleton_layer].unsqueeze(3)
    chunks = [torch.full((2, 2, 4, 24), -1.0)]
    desc = PageBufferShapeDesc()
    desc.nl, desc.nb, desc.bs = 2, 4, 2
    desc.nh, desc.hs, desc.kv_size = 3, 8, 2
    desc.element_size, desc.dtype = 4, torch.float32
    with pytest.raises(NotImplementedError, match="rank-5 paged tensors"):
        MluDeviceOps().multi_layer_block_kv_transfer(
            pages,
            chunks,
            [0, 1],
            "cpu",
            direction,
            desc,
            4,
            EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS,
            0,
        )
    assert all(torch.equal(layer, torch.ones_like(layer)) for layer in pages)
    assert torch.equal(chunks[0], torch.full_like(chunks[0], -1))


@pytest.mark.parametrize("direction", [TransferDirection.H2D, TransferDirection.D2H])
def test_tensor_copy_and_pointer_rejection(direction: TransferDirection) -> None:
    """Tensor copies honor byte count; even valid host pointers are rejected."""
    ops = MluDeviceOps()
    source = torch.arange(12, dtype=torch.float32)
    dest = torch.full_like(source, -1)
    ops.lmcache_memcpy_async(dest, source, 20, direction, 0, 1)
    assert torch.equal(dest[:5], source[:5])
    assert torch.equal(dest[5:], torch.full_like(dest[5:], -1))
    for dst, src in ((dest.data_ptr(), source.data_ptr()), (dest, source.data_ptr())):
        with pytest.raises(TypeError, match="tensor operands"):
            ops.lmcache_memcpy_async(dst, src, 20, direction, 0, 1)


@pytest.mark.parametrize("nbytes", [-1, 3, 1000])
def test_copy_rejects_invalid_size(nbytes: int) -> None:
    """Negative, unaligned, and oversized copies cannot silently truncate."""
    tensor = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError):
        MluDeviceOps().lmcache_memcpy_async(
            tensor, tensor, nbytes, TransferDirection.H2D, 0, 1
        )


def test_inprocess_transfer_is_explicitly_unsupported() -> None:
    """Legacy token transfers cannot silently use the NHD fallback on HND data."""
    ops = MluDeviceOps()
    with pytest.raises(NotImplementedError, match="engine_driven"):
        ops.multi_layer_kv_transfer()
    with pytest.raises(NotImplementedError, match="engine_driven"):
        ops.single_layer_kv_transfer()
