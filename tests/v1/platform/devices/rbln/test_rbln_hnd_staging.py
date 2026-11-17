# SPDX-License-Identifier: Apache-2.0
"""HND requirements, allocation lifetime and concurrent transfer contracts."""

# Standard
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any
import gc
import threading
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.platform.devices.rbln import staging
from lmcache.v1.platform.devices.rbln.device_ops import RblnDeviceOps
from lmcache.v1.platform.devices.rbln.staging import hnd_staging
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops import multi_layer_block_kv_transfer
import lmcache.lmcache_native as lmcache_native

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection
pytestmark = pytest.mark.usefixtures("cpu_rbln")


def _operands(fill: float = 1) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    layers = [
        torch.full((8, 2, 2, 1, 4, 64), fill + i, dtype=torch.bfloat16)
        for i in range(3)
    ]
    return layers, [torch.full((2, 3, 8, 128), -1, dtype=torch.bfloat16)]


def _desc(layers: int = 3) -> PageBufferShapeDesc:
    desc = PageBufferShapeDesc()
    desc.kv_size, desc.nl, desc.nb = 2, layers, 8
    desc.nh, desc.bs, desc.hs, desc.element_size = 2, 4, 64, 2
    return desc


def _transfer(
    layers: list[torch.Tensor], chunks: list[torch.Tensor], direction: TransferDirection
) -> None:
    RblnDeviceOps().multi_layer_block_kv_transfer(
        layers,
        chunks,
        [5, 2],
        layers[0].device,
        direction,
        _desc(),
        8,
        EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS,
        0,
    )


@pytest.mark.parametrize("layout", ["blocks_first", "kv_first"])
@pytest.mark.parametrize(
    "direction,skip",
    [
        (TransferDirection.D2H, 0),
        (TransferDirection.H2D, 0),
        (TransferDirection.H2D, 1),
        (TransferDirection.H2D, 3),
        (TransferDirection.H2D, 7),
    ],
)
@pytest.mark.parametrize("bpc", [1, 3])
@pytest.mark.parametrize("chiplets", [1, 3])
def test_hnd_bytes_match_canonical_path(
    cpu_rbln: Any,
    layout: str,
    direction: TransferDirection,
    bpc: int,
    skip: int,
    chiplets: int,
) -> None:
    """Both directions independently match canonical bytes, including untouched tails.

    Odd layer/shard counts exercise a shard spanning the K/V boundary. All
    chunks start with sentinel bytes; block IDs are permuted and the final
    multi-block chunk is partial. A fresh worker owns each transfer.
    """
    cpu_rbln.chiplets = chiplets
    torch.manual_seed(18)
    layers = [torch.randn((8, 2, 2, 1, 4, 64), dtype=torch.bfloat16) for _ in range(5)]
    if layout == "kv_first":
        layers = [layer.transpose(0, 1).contiguous() for layer in layers]
        fmt = EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS
        canonical_fmt = EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    else:
        fmt = EngineKVFormat.NL_X_NB_TWO_NH_ONE_BS_HS
        canonical_fmt = EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
    expected_layers = [layer.clone() for layer in layers]
    chunks = [
        torch.randn((2, 5, bpc * 4, 128), dtype=torch.bfloat16)
        for _ in range((7 + bpc - 1) // bpc)
    ]
    expected_chunks = [chunk.clone() for chunk in chunks]
    ids = [5, 2, 7, 1, 6, 0, 3]
    args = (ids, torch.device("cpu"), direction, _desc(5), bpc * 4)
    multi_layer_block_kv_transfer(
        [layer.squeeze(3) for layer in expected_layers],
        expected_chunks,
        *args,
        canonical_fmt,
        skip,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(
            RblnDeviceOps().multi_layer_block_kv_transfer,
            layers,
            chunks,
            *args,
            fmt,
            skip,
        ).result(timeout=10)
    for got, expected in zip(
        layers + chunks, expected_layers + expected_chunks, strict=True
    ):
        assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
@pytest.mark.parametrize(
    "missing,match",
    [
        ("kv_dtype", "bf16"),
        ("chunk_dtype", "bf16"),
        ("head_size", "multiple of 64"),
        ("compiled_copy", "copy_strided_view.*PrivateUse1"),
    ],
)
def test_requirements_fail_before_copy(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    match: str,
    direction: TransferDirection,
) -> None:
    """Each missing premise raises while every input/output byte stays unchanged."""
    layers, chunks = _operands()
    if missing == "kv_dtype":
        layers = [layer.float() for layer in layers]
    elif missing == "chunk_dtype":
        chunks = [chunk.float() for chunk in chunks]
    elif missing == "head_size":
        layers = [layer[..., :32].contiguous() for layer in layers]
    elif missing == "compiled_copy":
        monkeypatch.setattr(
            torch._C, "_dispatch_has_kernel_for_dispatch_key", lambda *args: False
        )
    before = [tensor.clone() for tensor in layers + chunks]

    def forbidden_copy(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("copy reached before requirement validation")

    monkeypatch.setattr(torch, "_foreach_copy_", forbidden_copy)
    error = RuntimeError if missing == "compiled_copy" else ValueError
    with pytest.raises(error, match=match):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(_transfer, layers, chunks, direction).result(timeout=10)
    assert all(
        torch.equal(got, old) for got, old in zip(layers + chunks, before, strict=True)
    )


@pytest.mark.parametrize("missing", ["set_device_alloc", "DevicePlacement"])
def test_missing_placement_uses_one_shard(
    cpu_rbln: Any,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    missing: str,
) -> None:
    """Either missing placement API permits canonical transfers and warns once."""
    monkeypatch.delattr(torch.rbln, missing)
    monkeypatch.delattr(torch.rbln, "get_device_properties")
    monkeypatch.setattr(
        staging.logger, "handlers", [*staging.logger.handlers, caplog.handler]
    )

    def no_chiplet_query(device: torch.device) -> int:
        raise AssertionError("unplaced staging must not query chiplets")

    monkeypatch.setattr(staging, "chiplet_count", no_chiplet_query)
    layers, chunks = _operands(11)
    expected_chunks = [torch.empty_like(chunk) for chunk in chunks]
    expected_layers = [torch.zeros_like(layer) for layer in layers]
    args = ([5, 2], torch.device("cpu"))
    fmt = EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
    multi_layer_block_kv_transfer(
        [layer.squeeze(3) for layer in layers],
        expected_chunks,
        *args,
        TransferDirection.D2H,
        _desc(),
        8,
        fmt,
        0,
    )
    multi_layer_block_kv_transfer(
        [layer.squeeze(3) for layer in expected_layers],
        expected_chunks,
        *args,
        TransferDirection.H2D,
        _desc(),
        8,
        fmt,
        0,
    )
    restored = [torch.zeros_like(layer) for layer in layers]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_transfer, layers, chunks, TransferDirection.D2H).result(timeout=10)
        pool.submit(_transfer, restored, chunks, TransferDirection.H2D).result(
            timeout=10
        )
    assert all(
        torch.equal(got, expected)
        for got, expected in zip(
            chunks + restored, expected_chunks + expected_layers, strict=True
        )
    )
    restored_set = hnd_staging(torch.device("cpu"), torch.bfloat16, 3, 2, 4, 64)
    assert len(restored_set.head_major) == len(restored_set.token_major) == 1
    assert cpu_rbln.placements == []
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == staging.__name__ and record.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert f"torch.rbln.{missing}" in warnings[0]
    assert "staging is not spread over chiplets" in warnings[0]


@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
def test_first_transfer_reserves_restore_memory_and_recovers(
    cpu_rbln: Any, direction: TransferDirection
) -> None:
    """Insufficient restore memory fails before either direction copies bytes."""
    layers, chunks = _operands()
    chunks[0].fill_(37)
    before = [tensor.clone() for tensor in layers + chunks]
    needed = 2 * 2 * 3 * 2 * 4 * 64 * 2
    cpu_rbln.byte_limit = needed - 2
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="placement capacity"):
            pool.submit(_transfer, layers, chunks, direction).result(timeout=10)
        assert all(
            torch.equal(got, old)
            for got, old in zip(layers + chunks, before, strict=True)
        )
        cpu_rbln.byte_limit = None
        pool.submit(_transfer, layers, chunks, direction).result(timeout=10)
    restored_set = hnd_staging(torch.device("cpu"), torch.bfloat16, 3, 2, 4, 64)
    buffers = restored_set.head_major + restored_set.token_major
    assert len({tensor.data_ptr() for tensor in buffers}) == 6
    assert sum(tensor.numel() * tensor.element_size() for tensor in buffers) == needed
    for layer in range(3):
        if direction == TransferDirection.D2H:
            assert torch.all(chunks[0][:, layer] == 1 + layer)
        else:
            assert torch.all(layers[layer][[5, 2]] == 37)


def test_store_does_not_submit_a_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stores succeed with canonical bytes when a device swap would refuse to run."""
    layers, chunks = _operands(11)
    original = torch.Tensor.copy_

    def copy(
        dst: torch.Tensor, src: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        if src.ndim == 4 and not src.is_contiguous():
            raise RuntimeError("device swap is unavailable on this thread")
        return original(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_transfer, layers, chunks, TransferDirection.D2H).result(timeout=10)
    for layer in range(3):
        assert torch.all(chunks[0][:, layer] == 11 + layer)


def test_cpu_reference_ignores_device_memory_apis(
    cpu_rbln: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CPU references work even when the backend's device allocators reject CPU."""
    cpu_rbln.chiplets = 1

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("device allocator does not accept CPU tensors")

    monkeypatch.setattr(torch.rbln, "set_device_alloc", forbidden)
    monkeypatch.setattr(torch.rbln, "bind_device_memory", forbidden, raising=False)
    layers, chunks = _operands(11)
    restored = [torch.zeros_like(layer) for layer in layers]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_transfer, layers, chunks, TransferDirection.D2H).result(timeout=10)
        pool.submit(_transfer, restored, chunks, TransferDirection.H2D).result(
            timeout=10
        )
    for layer in range(3):
        assert torch.all(chunks[0][:, layer] == 11 + layer)
        assert torch.all(restored[layer][[5, 2]] == 11 + layer)


def test_only_latest_geometry_is_kept() -> None:
    """A new model geometry releases obsolete restore buffers."""
    first = hnd_staging(torch.device("cpu"), torch.bfloat16, 3, 2, 4, 64)
    refs = [weakref.ref(tensor) for tensor in first.head_major + first.token_major]
    assert hnd_staging(torch.device("cpu"), torch.bfloat16, 3, 2, 4, 64) is first
    del first
    hnd_staging(torch.device("cpu"), torch.bfloat16, 5, 2, 4, 64)
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_store_and_restore_do_not_share_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A host-staged store and a device-staged restore preserve independent bytes."""
    store_layers, stored = _operands(11)
    restored, source = _operands(31)
    source[0].fill_(37)
    rendezvous = threading.Barrier(2)
    first_leg = threading.local()
    original = torch._foreach_copy_

    def interleave(dst: list[torch.Tensor], src: list[torch.Tensor]) -> None:
        original(dst, src)
        if not getattr(first_leg, "done", False):
            first_leg.done = True
            rendezvous.wait(timeout=10)

    monkeypatch.setattr(torch, "_foreach_copy_", interleave)
    with ThreadPoolExecutor(max_workers=2) as pool:
        store = pool.submit(_transfer, store_layers, stored, TransferDirection.D2H)
        restore = pool.submit(_transfer, restored, source, TransferDirection.H2D)
        store.result(timeout=15)
        restore.result(timeout=15)
    for layer in range(3):
        assert torch.all(stored[0][:, layer] == 11 + layer)
        assert torch.all(restored[layer][[5, 2]] == 37)
        assert torch.all(restored[layer][[0, 1, 3, 4, 6, 7]] == 31 + layer)


def test_restores_serialize_the_whole_transfer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A later restore cannot reuse the set while an earlier one holds its bytes."""
    first_layers, first_chunks = _operands(11)
    second_layers, second_chunks = _operands(31)
    first_chunks[0].fill_(17)
    second_chunks[0].fill_(37)
    paused, release, second_started = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    local = threading.local()
    original = torch._foreach_copy_

    def pause_first(dst: list[torch.Tensor], src: list[torch.Tensor]) -> None:
        original(dst, src)
        if getattr(local, "first", False) and not paused.is_set():
            paused.set()
            if not release.wait(timeout=10):
                raise RuntimeError("test did not release first restore")

    def restore_first() -> None:
        local.first = True
        _transfer(first_layers, first_chunks, TransferDirection.H2D)

    def restore_second() -> None:
        second_started.set()
        _transfer(second_layers, second_chunks, TransferDirection.H2D)

    monkeypatch.setattr(torch, "_foreach_copy_", pause_first)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(restore_first)
        assert paused.wait(timeout=10)
        second = pool.submit(restore_second)
        try:
            assert second_started.wait(timeout=10)
            with pytest.raises(TimeoutError):
                second.result(timeout=0.05)
        finally:
            release.set()
        first.result(timeout=10)
        second.result(timeout=10)
    for layer in range(3):
        assert torch.all(first_layers[layer][[5, 2]] == 17)
        assert torch.all(second_layers[layer][[5, 2]] == 37)
        assert torch.all(first_layers[layer][[0, 1, 3, 4, 6, 7]] == 11 + layer)
        assert torch.all(second_layers[layer][[0, 1, 3, 4, 6, 7]] == 31 + layer)


def test_parallel_stores_keep_host_staging_per_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four stores paused after gathering still publish their own canonical bytes."""
    gate = threading.Barrier(4)
    original = torch._foreach_copy_

    def interleave(dst: list[torch.Tensor], src: list[torch.Tensor]) -> None:
        original(dst, src)
        gate.wait(timeout=10)

    def store(fill: int) -> list[torch.Tensor]:
        layers, chunks = _operands(fill)
        _transfer(layers, chunks, TransferDirection.D2H)
        return chunks

    monkeypatch.setattr(torch, "_foreach_copy_", interleave)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(fill, pool.submit(store, fill)) for fill in (11, 21, 31, 41)]
        for fill, future in futures:
            chunks = future.result(timeout=15)
            for layer in range(3):
                assert torch.all(chunks[0][:, layer] == fill + layer)
