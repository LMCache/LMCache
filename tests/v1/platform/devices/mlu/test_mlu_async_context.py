# SPDX-License-Identifier: Apache-2.0
"""Real MLU ordering and ownership through the async worker context."""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.async_engine_driven import (
    AsyncEngineDrivenTransferContext,
)
from lmcache.v1.platform.devices.mlu import MluDeviceSpec

pytestmark = pytest.mark.no_shared_allocator


@pytest.mark.parametrize("shm", [False, True])
def test_async_store_flush_and_close_on_mlu(
    monkeypatch: pytest.MonkeyPatch, shm: bool
) -> None:
    """Concurrent stores retain sources and commit only completed host data."""
    if not MluDeviceSpec().is_available():
        pytest.skip("requires torch_mlu and an MLU device")
    kv = {str(i): torch.zeros(2, 4, 2, 4, 8, device="mlu") for i in range(2)}
    chunk_shape = (2, 2, 16, 16)
    transport = MagicMock()
    transport.layout_desc = SimpleNamespace(
        shapes=[torch.Size(chunk_shape)], dtypes=[torch.float32]
    )
    transport.prepare_store.side_effect = (
        (lambda *a: ([torch.empty(chunk_shape)], [0])) if shm else (lambda *a: None)
    )
    seen: set[int] = set()
    lock = threading.Lock()

    def commit(key: int, instance_id: int, chunks: list[torch.Tensor]) -> bool:
        assert instance_id == 1
        assert len(chunks) == 1
        assert torch.equal(chunks[0], torch.full(chunk_shape, key, dtype=torch.float32))
        with lock:
            seen.add(key)
        return True

    transport.commit_store.side_effect = commit
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: transport
    )
    ctx = worker_transfer.create_transfer_context(
        kv, instance_id=1, req_client=MagicMock(), mode="engine_driven"
    )
    assert isinstance(ctx, AsyncEngineDrivenTransferContext)
    ctx.register(kv, "test", 1, 4, 1.0, layout_hints={"kv_layout": "HND"})
    producer = torch.mlu.Stream()
    futures = []
    try:
        for value in range(1, 9):
            with torch.mlu.stream(producer):
                source = {
                    key: torch.full_like(layer, value) for key, layer in kv.items()
                }
                event = ctx.create_recorded_event()
            futures.append(
                ctx.submit_store(str(value), value, source, [[3, 1, 0, 2]], event, 4)
            )
            del source, event
        ctx.flush_inflight_stores()
        ctx.close()
        assert all(f.result(timeout=2) for f in futures)
        with lock:
            assert seen == set(range(1, 9))
        transport.close.assert_called_once()
    finally:
        ctx.close()


@pytest.mark.parametrize("pin_memory", [False, True], ids=["pageable", "pinned"])
@pytest.mark.parametrize("prefix_tokens", [192, 256])
def test_fractional_kv_roundtrip_and_immediate_host_reuse(
    monkeypatch: pytest.MonkeyPatch, pin_memory: bool, prefix_tokens: int
) -> None:
    """Completion preserves fp16 bits across streams before recycling host data.

    Match Qwen's 24-layer HND geometry, use distinct source/destination block
    orders, and compare every bit including signed zero and subnormals. Only
    the public transfer futures order consumers; the test never adds a device
    synchronization before consuming committed chunks or recycling host slots.
    """
    if not MluDeviceSpec().is_available():
        pytest.skip("requires torch_mlu and an MLU device")
    shape = (2, 20, 2, 16, 64)
    chunk_shape = (2, 24, 64, 128)
    num_blocks = prefix_tokens // 16
    source_ids = list(reversed(range(num_blocks)))
    target_ids = list(range(2, num_blocks + 2))
    chunks: list[torch.Tensor] = []
    expected_chunks: list[torch.Tensor] = []
    transport = MagicMock()
    transport.layout_desc = SimpleNamespace(
        shapes=[torch.Size(chunk_shape)], dtypes=[torch.float16]
    )

    def prepare_store(*args: object) -> tuple[list[torch.Tensor], list[int]]:
        return (
            [
                torch.empty(chunk_shape, dtype=torch.float16, pin_memory=pin_memory)
                for _ in range(prefix_tokens // 64)
            ],
            list(range(prefix_tokens // 64)),
        )

    def commit_store(key: int, instance_id: int, completed: list[torch.Tensor]) -> bool:
        for actual, expected in zip(completed, expected_chunks, strict=True):
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        chunks[:] = completed
        return True

    def commit_retrieve(*args: object) -> bool:
        # Simulate the server immediately reusing SHM slots at the release ACK.
        for chunk in chunks:
            chunk.fill_(42)
        return True

    transport.prepare_store.side_effect = prepare_store
    transport.commit_store.side_effect = commit_store
    transport.prepare_retrieve.side_effect = lambda *args: chunks
    transport.commit_retrieve.side_effect = commit_retrieve
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: transport
    )
    initial = {
        str(i): torch.zeros(shape, device="mlu", dtype=torch.float16) for i in range(24)
    }
    ctx = worker_transfer.create_transfer_context(
        initial, instance_id=1, req_client=MagicMock(), mode="engine_driven"
    )
    ctx.register(initial, "fractional-kv", 1, 4, 1.0, layout_hints={"kv_layout": "HND"})
    producer, restore_stream, consumer = (torch.mlu.Stream() for _ in range(3))
    rng = torch.Generator().manual_seed(42)
    try:
        for iteration in range(8):
            host = {
                str(i): torch.randn(shape, generator=rng, dtype=torch.float32).half()
                for i in range(24)
            }
            for layer in host.values():
                layer.flatten()[:4] = torch.tensor(
                    [-0.0, 0.0, 2**-24, -(2**-24)], dtype=torch.float16
                )
            canonical = torch.stack(
                [
                    layer[:, source_ids]
                    .permute(0, 1, 3, 2, 4)
                    .reshape(2, prefix_tokens, 128)
                    for layer in host.values()
                ],
                dim=1,
            )
            expected_chunks[:] = [
                chunk.contiguous() for chunk in canonical.split(64, dim=2)
            ]
            with torch.mlu.stream(producer):
                source = {
                    name: layer.to("mlu", non_blocking=True)
                    for name, layer in host.items()
                }
                event = ctx.create_recorded_event()
            future = ctx.submit_store(
                str(iteration), iteration, source, [source_ids], event, 4
            )
            del source, event
            assert future.result(timeout=30)
            with torch.mlu.stream(restore_stream):
                target = {
                    name: torch.full(shape, -9, device="mlu", dtype=torch.float16)
                    for name in host
                }
                retrieved = ctx.submit_retrieve(
                    str(iteration),
                    iteration,
                    target,
                    [target_ids],
                    ctx.create_recorded_event(),
                    4,
                )
                assert retrieved.result(timeout=30)
            with torch.mlu.stream(consumer):
                observed = {name: layer.clone() for name, layer in target.items()}
            consumer.synchronize()
            for name, layer in observed.items():
                reference = torch.full_like(host[name], -9)
                reference[:, target_ids] = host[name][:, source_ids]
                assert torch.equal(
                    layer.cpu().view(torch.int16), reference.view(torch.int16)
                )
        ctx.flush_inflight_stores()
    finally:
        ctx.close()


@pytest.mark.parametrize("shm", [False, True], ids=["inline", "shm"])
def test_preemption_reuses_blocks_while_commit_and_close_are_pending(
    monkeypatch: pytest.MonkeyPatch, shm: bool
) -> None:
    """Flush protects the source; close protects queued commits and their buffers."""
    if not MluDeviceSpec().is_available():
        pytest.skip("requires torch_mlu and an MLU device")
    shape = (2, 8, 2, 16, 64)
    chunk_shape = (2, 4, 64, 128)
    block_ids = [7, 2, 5, 0]
    rng = torch.Generator().manual_seed(167)
    versions = [
        {str(i): torch.randn(shape, generator=rng).half() for i in range(4)}
        for _ in range(2)
    ]
    expected = [
        torch.stack(
            [
                layer[:, block_ids].permute(0, 1, 3, 2, 4).reshape(2, 64, 128)
                for layer in version.values()
            ],
            dim=1,
        )
        for version in versions
    ]
    prepare_entered = threading.Event()
    release_prepare = threading.Event()
    commit_entered = threading.Event()
    release_commit = threading.Event()
    flush_entered = threading.Event()
    flush_done = threading.Event()
    close_entered = threading.Event()
    close_done = threading.Event()
    errors: list[BaseException] = []
    committed: list[int] = []
    transport = MagicMock()
    transport.layout_desc = SimpleNamespace(
        shapes=[torch.Size(chunk_shape)], dtypes=[torch.float16]
    )

    def prepare(
        key: int, instance_id: int
    ) -> tuple[list[torch.Tensor], list[int]] | None:
        if key == 0:
            prepare_entered.set()
            assert release_prepare.wait(30)
        if shm:
            return [torch.empty(chunk_shape, dtype=torch.float16)], [0]
        return None

    def commit(key: int, instance_id: int, chunks: list[torch.Tensor]) -> bool:
        commit_entered.set()
        assert release_commit.wait(30)
        assert len(chunks) == 1
        assert torch.equal(chunks[0].view(torch.int16), expected[key].view(torch.int16))
        committed.append(key)
        return True

    transport.prepare_store.side_effect = prepare
    transport.commit_store.side_effect = commit
    monkeypatch.setattr(
        worker_transfer, "create_engine_driven_context", lambda *a, **k: transport
    )
    kv = {
        name: torch.empty_like(layer, device="mlu")
        for name, layer in versions[0].items()
    }
    ctx = worker_transfer.create_transfer_context(
        kv, instance_id=1, req_client=MagicMock(), mode="engine_driven"
    )
    ctx.register(kv, "preemption-reuse", 1, 4, 1.0, layout_hints={"kv_layout": "HND"})
    producer = torch.mlu.Stream()

    def flush() -> None:
        flush_entered.set()
        try:
            ctx.flush_inflight_stores()
        except BaseException as exc:
            errors.append(exc)
        finally:
            flush_done.set()

    def close() -> None:
        close_entered.set()
        try:
            ctx.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            close_done.set()

    flushing = threading.Thread(target=flush)
    closing = threading.Thread(target=close)
    try:
        with torch.mlu.stream(producer):
            for name, layer in kv.items():
                layer.copy_(versions[0][name], non_blocking=True)
            first = ctx.submit_store(
                "old", 0, kv, [block_ids], ctx.create_recorded_event(), 4
            )
        assert prepare_entered.wait(10)
        assert not first.query()
        flushing.start()
        assert flush_entered.wait(10)
        assert not flush_done.wait(0.1)
        release_prepare.set()
        assert flush_done.wait(30)
        assert not errors
        assert commit_entered.wait(10)
        assert not first.query()
        # These are the same allocations and physical blocks, now owned by a
        # new request. The prior store must depend only on its retained host copy.
        with torch.mlu.stream(producer):
            for name, layer in kv.items():
                layer.copy_(versions[1][name], non_blocking=True)
            second = ctx.submit_store(
                "new", 1, kv, [block_ids], ctx.create_recorded_event(), 4
            )
        ctx.flush_inflight_stores()
        assert not second.query()
        del kv, layer, producer
        closing.start()
        assert close_entered.wait(10)
        assert not close_done.wait(0.1)
        transport.close.assert_not_called()
        release_commit.set()
        assert close_done.wait(30)
        assert not errors
        assert first.result(timeout=2) and second.result(timeout=2)
        assert committed == [0, 1]
        transport.close.assert_called_once()
    finally:
        release_prepare.set()
        release_commit.set()
        for thread in (flushing, closing):
            if thread.ident is not None:
                thread.join(timeout=35)
                assert not thread.is_alive()
        ctx.close()
