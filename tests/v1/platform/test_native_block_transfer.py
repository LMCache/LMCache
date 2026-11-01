# SPDX-License-Identifier: Apache-2.0
"""CUDA, XPU and Ascend serve tensor block transfers on an address kernel.

CPU tests install fake extension modules for the real CUDA, XPU and Ascend ops
classes and record what reaches the kernel. CUDA tests exercise the compiled
kernel end to end.
"""

# Standard
from dataclasses import dataclass
from types import ModuleType
from typing import cast
from unittest.mock import MagicMock
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
from lmcache.v1.gpu_connector.utils import (
    make_page_buffer_shape_desc,
    normalize_kv_and_discover_format,
)
from lmcache.v1.platform import resolve_device_ops
from lmcache.v1.platform.base.device_ops import DeviceOps
from lmcache.v1.platform.base.device_spec import DeviceSpec
from lmcache.v1.platform.devices.cuda import CudaDeviceSpec
from lmcache.v1.platform.devices.cuda import block_transfer as cuda_block_transfer
from lmcache.v1.platform.devices.cuda.device_ops import CudaDeviceOps
from lmcache.v1.platform.devices.npu import NpuDeviceSpec
from lmcache.v1.platform.devices.npu import block_transfer as npu_block_transfer
from lmcache.v1.platform.devices.npu.device_ops import NpuDeviceOps
from lmcache.v1.platform.devices.xpu import XpuDeviceSpec
from lmcache.v1.platform.devices.xpu import block_transfer as xpu_block_transfer
from lmcache.v1.platform.devices.xpu.device_ops import XpuDeviceOps
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
import lmcache.lmcache_native as lmcache_native

D2H = lmcache_native.TransferDirection.D2H
H2D = lmcache_native.TransferDirection.H2D
CPU = torch.device("cpu")
_BLOCK_SIZE = 4
_BLOCKS_PER_CHUNK = 2
_CHUNK_TOKENS = _BLOCK_SIZE * _BLOCKS_PER_CHUNK

_NATIVE_BACKENDS = [
    pytest.param(CudaDeviceOps, "lmcache.cuda_ops", id="cuda"),
    pytest.param(XpuDeviceOps, "lmcache.xpu_ops", id="xpu"),
    pytest.param(NpuDeviceOps, "lmcache_ascend.ops", id="npu"),
]
_NATIVE_SPECS = [
    pytest.param(CudaDeviceSpec, "lmcache.cuda_ops", id="cuda"),
    pytest.param(XpuDeviceSpec, "lmcache.xpu_ops", id="xpu"),
    pytest.param(NpuDeviceSpec, "lmcache_ascend.ops", id="npu"),
]
# NPU requires its plugin; only CUDA and XPU run on the baseline without one.
_OPTIONAL_EXTENSION_BACKENDS = _NATIVE_BACKENDS[:2]
# Each backend owns its lowering, including the device sync it issues.
_BLOCK_TRANSFER_MODULES: dict[type[DeviceOps], ModuleType] = {
    CudaDeviceOps: cuda_block_transfer,
    XpuDeviceOps: xpu_block_transfer,
    NpuDeviceOps: npu_block_transfer,
}


@dataclass
class _Launch:
    """Operands one kernel launch received."""

    paged: torch.Tensor
    chunks: list[int]
    block_ids: torch.Tensor
    direction: lmcache_native.TransferDirection
    skip: int


class _RecordingKernel:
    """Stand-in for a compiled address-taking block-transfer kernel."""

    def __init__(self) -> None:
        self.launches: list[_Launch] = []

    def __call__(
        self,
        paged: torch.Tensor,
        chunks: list[int],
        block_ids: torch.Tensor,
        device: torch.device,
        direction: lmcache_native.TransferDirection,
        shape_desc: PageBufferShapeDesc,
        chunk_size: int,
        engine_kv_format: lmcache_native.EngineKVFormat,
        skip: int,
    ) -> None:
        self.launches.append(_Launch(paged, list(chunks), block_ids, direction, skip))


def _install_native(
    monkeypatch: pytest.MonkeyPatch, module_name: str, **symbols: object
) -> None:
    """Make ``import <module_name>`` load a fake module exporting ``symbols``."""
    parent_name, _, child_name = module_name.rpartition(".")
    module = ModuleType(module_name)
    for name, value in symbols.items():
        setattr(module, name, value)
    parent = sys.modules.get(parent_name) or ModuleType(parent_name)
    monkeypatch.setitem(sys.modules, parent_name, parent)
    monkeypatch.setattr(parent, child_name, module, raising=False)
    monkeypatch.setitem(sys.modules, module_name, module)


def _native_ops(
    monkeypatch: pytest.MonkeyPatch, ops_cls: type[DeviceOps], module_name: str
) -> tuple[DeviceOps, _RecordingKernel]:
    """Return ``ops_cls`` bound to a recording kernel."""
    kernel = _RecordingKernel()
    _install_native(monkeypatch, module_name, multi_layer_block_kv_transfer=kernel)
    ops = ops_cls()
    ops.ensure_native()
    return ops, kernel


def _paged_kv(
    num_blocks: int, device: str = "cpu", dtype: torch.dtype = torch.float32
) -> tuple[DiscoverableKVCache, lmcache_native.EngineKVFormat, PageBufferShapeDesc]:
    """Two NHD layers ``[2, num_blocks, 4, 2, 8]`` with their format and shape."""
    layers = [
        (
            torch.arange(2 * num_blocks * _BLOCK_SIZE * 16).reshape(
                2, num_blocks, _BLOCK_SIZE, 2, 8
            )
            + 1000 * i
        ).to(device=device, dtype=dtype)
        for i in range(2)
    ]
    fmt, normalized = normalize_kv_and_discover_format(
        layers, EngineType.VLLM, layout_hints={"kv_layout": "NHD"}
    )
    shape_desc = make_page_buffer_shape_desc(
        normalized,
        fmt,
        layer_idx=0,
        num_layers_in_group=2,
        num_blocks=num_blocks,
        block_size=_BLOCK_SIZE,
    )
    return normalized, fmt, shape_desc


def _layers(paged: DiscoverableKVCache) -> list[torch.Tensor]:
    """Return the per-layer tensors of an NHD ``paged`` KV list."""
    assert isinstance(paged, list)
    return [layer for layer in paged if isinstance(layer, torch.Tensor)]


def _expected_chunk(paged: DiscoverableKVCache, blocks: list[int]) -> torch.Tensor:
    """Canonical ``[2, L, T, H*D]`` chunk holding ``blocks`` of every layer."""
    return torch.stack(
        [
            layer.cpu()[:, blocks].reshape(2, len(blocks) * _BLOCK_SIZE, 16)
            for layer in _layers(paged)
        ],
        dim=1,
    )


def _host_chunk(address: int, pinned: bool = True) -> MagicMock:
    """Host chunk stand-in; real pinning needs an accelerator.

    ``spec=torch.Tensor`` makes ``isinstance(chunk, torch.Tensor)`` true, so the
    op takes its tensor path rather than the legacy address path.
    """
    chunk = MagicMock(spec=torch.Tensor)
    chunk.is_pinned.return_value = pinned
    chunk.data_ptr.return_value = address
    return chunk


# -- CPU: fake extension modules -------------------------------------------


@pytest.mark.parametrize(("ops_cls", "module_name"), _NATIVE_BACKENDS)
def test_extension_without_kernel_fails_initialization(
    monkeypatch: pytest.MonkeyPatch, ops_cls: type[DeviceOps], module_name: str
) -> None:
    """An installed extension lacking the block kernel is a broken install.

    It must not quietly serve the torch baseline, and a second attempt must
    fail the same way instead of returning as already initialized.
    """
    _install_native(monkeypatch, module_name, calculate_cdf=lambda *_: None)
    ops = ops_cls()

    for _ in range(2):
        with pytest.raises(AttributeError, match="multi_layer_block_kv_transfer"):
            ops.ensure_native()


@pytest.mark.parametrize(("spec_cls", "module_name"), _NATIVE_SPECS)
def test_failed_initialization_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, spec_cls: type[DeviceSpec], module_name: str
) -> None:
    """Resolving a spec over a broken extension fails on every call."""
    _install_native(monkeypatch, module_name, calculate_cdf=lambda *_: None)
    spec = spec_cls()

    for _ in range(2):
        with pytest.raises(AttributeError, match="multi_layer_block_kv_transfer"):
            spec.get_ops()


@pytest.mark.parametrize(("ops_cls", "module_name"), _OPTIONAL_EXTENSION_BACKENDS)
def test_missing_extension_tensor_transfer_uses_baseline(
    monkeypatch: pytest.MonkeyPatch, ops_cls: type[DeviceOps], module_name: str
) -> None:
    """Without the CUDA or XPU extension, the torch baseline serves, byte-exact."""
    monkeypatch.setitem(sys.modules, module_name, None)
    ops = ops_cls()
    ops.ensure_native()
    paged, fmt, desc = _paged_kv(num_blocks=6)
    chunk = torch.zeros(2, 2, _CHUNK_TOKENS, 16)

    ops.multi_layer_block_kv_transfer(
        paged, [chunk], [1, 3], CPU, D2H, desc, _CHUNK_TOKENS, fmt, 0
    )

    assert torch.equal(chunk, _expected_chunk(paged, [1, 3]))


@pytest.mark.parametrize(("ops_cls", "module_name"), _OPTIONAL_EXTENSION_BACKENDS)
def test_missing_extension_output_allocation_is_plain(
    monkeypatch: pytest.MonkeyPatch, ops_cls: type[DeviceOps], module_name: str
) -> None:
    """Without the extension, block-transfer output is an ordinary CPU tensor."""
    monkeypatch.setitem(sys.modules, module_name, None)
    ops = ops_cls()
    ops.ensure_native()

    out = ops.allocate_block_transfer_output((2, 3), torch.float16)

    assert (out.shape, out.dtype, out.device) == ((2, 3), torch.float16, CPU)
    assert not out.is_pinned()


@pytest.mark.parametrize(("ops_cls", "module_name"), _NATIVE_BACKENDS)
@pytest.mark.parametrize("direction", [D2H, H2D])
def test_tensor_operands_lower_to_kernel_launches(
    monkeypatch: pytest.MonkeyPatch,
    ops_cls: type[DeviceOps],
    module_name: str,
    direction: lmcache_native.TransferDirection,
) -> None:
    """The public op keeps its tensor API after binding and lowers to the kernel."""
    ops, kernel = _native_ops(monkeypatch, ops_cls, module_name)
    paged, fmt, desc = _paged_kv(num_blocks=10)
    addresses = [0x1000 * (i + 1) for i in range(5)]
    chunks = cast(list[torch.Tensor], [_host_chunk(a) for a in addresses])

    ops.multi_layer_block_kv_transfer(
        paged, chunks, list(range(10)), CPU, direction, desc, _CHUNK_TOKENS, fmt, 1
    )

    layer_addresses = [layer.data_ptr() for layer in _layers(paged)]
    assert [launch.chunks for launch in kernel.launches] == [
        addresses[:4],
        addresses[4:],
    ]
    assert [launch.block_ids.tolist() for launch in kernel.launches] == [
        list(range(8)),
        [8, 9],
    ]
    # A skip inside the first launch leaves the later launches whole.
    assert [launch.skip for launch in kernel.launches] == [1, 0]
    for launch in kernel.launches:
        assert launch.paged.dtype == torch.int64
        assert launch.paged.tolist() == layer_addresses
        assert launch.block_ids.dtype == torch.int64
        assert launch.direction == direction


@pytest.mark.parametrize(("ops_cls", "module_name"), _NATIVE_BACKENDS)
def test_skip_spanning_launches_carries_into_later_launches(
    monkeypatch: pytest.MonkeyPatch, ops_cls: type[DeviceOps], module_name: str
) -> None:
    """A skip past the first launch's blocks continues into the next launch."""
    ops, kernel = _native_ops(monkeypatch, ops_cls, module_name)
    paged, fmt, desc = _paged_kv(num_blocks=10)
    chunks = cast(list[torch.Tensor], [_host_chunk(0x1000 * (i + 1)) for i in range(5)])

    ops.multi_layer_block_kv_transfer(
        paged, chunks, list(range(10)), CPU, H2D, desc, _CHUNK_TOKENS, fmt, 9
    )

    # Launch 1 holds blocks 0-7 (all skipped); launch 2 holds 8-9 (one skipped).
    assert [launch.skip for launch in kernel.launches] == [8, 1]


@pytest.mark.parametrize(("ops_cls", "module_name"), _NATIVE_BACKENDS)
def test_legacy_address_operands_are_forwarded_unchanged(
    monkeypatch: pytest.MonkeyPatch, ops_cls: type[DeviceOps], module_name: str
) -> None:
    """LMCache-driven callers' address form reaches the kernel as passed."""
    ops, kernel = _native_ops(monkeypatch, ops_cls, module_name)
    _, fmt, desc = _paged_kv(num_blocks=8)
    paged_addresses = torch.tensor([11, 22], dtype=torch.int64)
    block_ids = torch.arange(8)

    # These callers already batch to the kernel's limit; nothing is re-split.
    ops.multi_layer_block_kv_transfer(
        paged_addresses, [1, 2, 3, 4], block_ids, CPU, H2D, desc, _CHUNK_TOKENS, fmt, 3
    )

    (launch,) = kernel.launches
    assert launch.paged is paged_addresses
    assert launch.chunks == [1, 2, 3, 4]
    assert launch.block_ids is block_ids
    assert launch.skip == 3


@pytest.mark.parametrize(("ops_cls", "module_name"), _NATIVE_BACKENDS)
def test_unpinned_h2d_chunks_sync_after_launch(
    monkeypatch: pytest.MonkeyPatch, ops_cls: type[DeviceOps], module_name: str
) -> None:
    """Chunks pinned into temporaries are synchronized before the op returns.

    The kernel reads the temporaries through raw addresses that torch's stream
    tracking cannot see, so returning without a sync would let the caching
    host allocator recycle them while the copy is still in flight.
    """
    ops, kernel = _native_ops(monkeypatch, ops_cls, module_name)
    syncs: list[int] = []
    monkeypatch.setattr(
        _BLOCK_TRANSFER_MODULES[ops_cls],
        "synchronize_device",
        lambda _device: syncs.append(len(kernel.launches)),
    )
    unpinned = _host_chunk(0, pinned=False)
    unpinned.pin_memory.return_value = _host_chunk(0x2000)
    paged, fmt, desc = _paged_kv(num_blocks=2)

    ops.multi_layer_block_kv_transfer(
        paged,
        cast(list[torch.Tensor], [unpinned]),
        [0, 1],
        CPU,
        H2D,
        desc,
        _CHUNK_TOKENS,
        fmt,
        0,
    )

    assert [launch.chunks for launch in kernel.launches] == [[0x2000]]
    assert syncs == [1], "expected exactly one sync, after the launch"


@pytest.mark.parametrize(("ops_cls", "module_name"), _NATIVE_BACKENDS)
def test_pinned_chunks_keep_the_async_path(
    monkeypatch: pytest.MonkeyPatch, ops_cls: type[DeviceOps], module_name: str
) -> None:
    """Already pinned chunks are passed as-is and nothing synchronizes."""
    ops, kernel = _native_ops(monkeypatch, ops_cls, module_name)
    syncs: list[torch.device] = []
    monkeypatch.setattr(
        _BLOCK_TRANSFER_MODULES[ops_cls], "synchronize_device", syncs.append
    )
    paged, fmt, desc = _paged_kv(num_blocks=2)

    ops.multi_layer_block_kv_transfer(
        paged,
        cast(list[torch.Tensor], [_host_chunk(0x3000)]),
        [0, 1],
        CPU,
        H2D,
        desc,
        _CHUNK_TOKENS,
        fmt,
        0,
    )

    assert [launch.chunks for launch in kernel.launches] == [[0x3000]]
    assert syncs == []


def test_binding_one_instance_leaves_others_on_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kernel is bound per instance; a fresh instance uses the baseline."""
    _, kernel = _native_ops(monkeypatch, CudaDeviceOps, "lmcache.cuda_ops")
    paged, fmt, desc = _paged_kv(num_blocks=6)
    chunk = torch.zeros(2, 2, _CHUNK_TOKENS, 16)

    CudaDeviceOps().multi_layer_block_kv_transfer(
        paged, [chunk], [1, 3], CPU, D2H, desc, _CHUNK_TOKENS, fmt, 0
    )

    assert kernel.launches == []
    assert torch.equal(chunk, _expected_chunk(paged, [1, 3]))


# -- CUDA: the compiled kernel ---------------------------------------------


@pytest.fixture
def cuda_ops() -> DeviceOps:
    """The resolved CUDA ops, skipping unless the compiled kernel is present."""
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA device")
    pytest.importorskip("lmcache.cuda_ops")
    return resolve_device_ops("cuda")


@pytest.mark.cuda
def test_cuda_output_allocation_is_pinned(cuda_ops: DeviceOps) -> None:
    out = cuda_ops.allocate_block_transfer_output((2, 3), torch.float16)
    assert out.device == CPU
    assert out.is_pinned()


@pytest.mark.cuda
def test_cuda_tensor_round_trip_across_launches(cuda_ops: DeviceOps) -> None:
    """Five chunks (two launches) gather and scatter back byte-exact."""
    device = torch.device("cuda")
    paged, fmt, desc = _paged_kv(num_blocks=10, device="cuda", dtype=torch.float16)
    chunk_shape = (2, 2, _CHUNK_TOKENS, 16)
    chunks = [
        cuda_ops.allocate_block_transfer_output(chunk_shape, torch.float16)
        for _ in range(5)
    ]
    block_ids = list(range(10))

    cuda_ops.multi_layer_block_kv_transfer(
        paged, chunks, block_ids, device, D2H, desc, _CHUNK_TOKENS, fmt, 0
    )
    torch.cuda.synchronize()
    for i, chunk in enumerate(chunks):
        assert torch.equal(chunk, _expected_chunk(paged, [2 * i, 2 * i + 1]))

    restored, _, _ = _paged_kv(num_blocks=10, device="cuda", dtype=torch.float16)
    for layer in _layers(restored):
        layer.zero_()
    cuda_ops.multi_layer_block_kv_transfer(
        restored, chunks, block_ids, device, H2D, desc, _CHUNK_TOKENS, fmt, 0
    )
    torch.cuda.synchronize()
    for source, target in zip(_layers(paged), _layers(restored), strict=True):
        assert torch.equal(source, target)


@pytest.mark.cuda
def test_cuda_unpinned_output_is_staged_in_place(cuda_ops: DeviceOps) -> None:
    """An unpinned output is filled in place through pinned staging."""
    paged, fmt, desc = _paged_kv(num_blocks=6, device="cuda", dtype=torch.float16)
    out = torch.zeros(2, 2, _CHUNK_TOKENS, 16, dtype=torch.float16).share_memory_()
    assert not out.is_pinned()

    cuda_ops.multi_layer_block_kv_transfer(
        paged, [out], [1, 3], torch.device("cuda"), D2H, desc, _CHUNK_TOKENS, fmt, 0
    )

    # Staging synchronizes before copying back, so no extra sync is needed.
    assert torch.equal(out, _expected_chunk(paged, [1, 3]))


@pytest.mark.cuda
def test_cuda_legacy_address_form_still_transfers(cuda_ops: DeviceOps) -> None:
    """The LMCache-driven address form reaches the kernel unchanged."""
    device = torch.device("cuda")
    paged, fmt, desc = _paged_kv(num_blocks=6, device="cuda", dtype=torch.float16)
    chunk = cuda_ops.allocate_block_transfer_output(
        (2, 2, _CHUNK_TOKENS, 16), torch.float16
    )
    paged_addresses = torch.tensor(
        [layer.data_ptr() for layer in _layers(paged)], dtype=torch.int64, device=device
    )
    block_ids = torch.tensor([1, 3], dtype=torch.int64, device=device)

    cuda_ops.multi_layer_block_kv_transfer(
        paged_addresses,
        [chunk.data_ptr()],
        block_ids,
        device,
        D2H,
        desc,
        _CHUNK_TOKENS,
        fmt,
        0,
    )
    torch.cuda.synchronize()

    assert torch.equal(chunk, _expected_chunk(paged, [1, 3]))
