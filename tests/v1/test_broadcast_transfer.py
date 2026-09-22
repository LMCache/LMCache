# SPDX-License-Identifier: Apache-2.0
"""Exercise bounded retrieval through its public transfer and buffer contracts."""

# Standard
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import timedelta
from pathlib import Path
from typing import Any
import json
import multiprocessing
import threading

# Third Party
import pytest
import torch
import torch.distributed as dist

# First Party
from lmcache.v1.broadcast_transfer import (
    BroadcastTransfer,
    RetrievalBuffer,
    RetrievalChunk,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

pytestmark = pytest.mark.no_shared_allocator


def make_object(tensors: list[torch.Tensor], *, grouped: bool = False) -> MemoryObj:
    """Pack typed groups into the same public format used by cache backends."""
    raw = torch.cat([tensor.reshape(-1).view(torch.uint8) for tensor in tensors])
    return TensorMemoryObj(
        raw,
        MemoryObjMetadata(
            shape=tensors[0].shape,
            dtype=tensors[0].dtype,
            address=0,
            phy_size=raw.numel(),
            ref_count=1,
            fmt=MemoryFormat.KV_MLA_FMT,
            shapes=[tensor.shape for tensor in tensors] if grouped else None,
            dtypes=[tensor.dtype for tensor in tensors] if grouped else None,
        ),
        None,
    )


def make_transfer(
    buffer: RetrievalBuffer | None,
    *,
    synchronize: Callable[[], None] = lambda: None,
    broadcast: Callable[[torch.Tensor, int], None] = lambda tensor, src: None,
) -> BroadcastTransfer:
    """Create a single-rank transport without mocking tensor operations."""
    return BroadcastTransfer(
        buffer,
        is_source=True,
        source_rank=0,
        broadcast=broadcast,
        broadcast_object=lambda obj, src: obj,
        agree=lambda ready: ready,
        synchronize=synchronize,
    )


@pytest.mark.parametrize("num_chunks", [1, 3, 100])
@pytest.mark.parametrize("chunk_tokens", [1, 7, 43])
@pytest.mark.parametrize("budget", [100, 256])
def test_retrieval_is_bounded_and_preserves_every_token(
    num_chunks: int, chunk_tokens: int, budget: int
) -> None:
    prefix = 3
    num_tokens = prefix + num_chunks * chunk_tokens
    expected = torch.arange(2 * num_tokens * 3, dtype=torch.float32).reshape(
        1, 2, num_tokens, 3
    )
    destination = torch.full_like(expected, -1)
    buffer = RetrievalBuffer.get(torch.device("cpu"), budget)
    allocations: set[int] = set()
    window_sizes: list[int] = []

    def broadcast(tensor: torch.Tensor, src: int) -> None:
        allocations.add(tensor.untyped_storage().data_ptr())
        window_sizes.append(tensor.numel())

    def chunks() -> Generator[RetrievalChunk, None, None]:
        for start in range(prefix, num_tokens, chunk_tokens):
            end = start + chunk_tokens
            source = make_object([expected[:, :, start:end]])
            yield source, start, end
            # Source ownership ends at the next iterator advance. Poison it to
            # catch a transfer that defers reads beyond this point.
            assert source.raw_tensor is not None
            source.raw_tensor.zero_()

    def write(objects: list[MemoryObj], starts: list[int], ends: list[int]) -> None:
        for obj, start, end in zip(objects, starts, ends, strict=True):
            assert obj.tensor is not None
            destination[:, :, start:end].copy_(obj.tensor)

    result = make_transfer(buffer, broadcast=broadcast).retrieve(
        chunks(), num_tokens, write
    )
    assert result.tolist() == [False] * prefix + [True] * (num_tokens - prefix)
    torch.testing.assert_close(destination[:, :, prefix:], expected[:, :, prefix:])
    assert (destination[:, :, :prefix] == -1).all()
    assert len(allocations) == 1
    assert max(window_sizes) <= budget
    if num_chunks * chunk_tokens * 24 > budget:
        assert len(window_sizes) > 1


@pytest.mark.parametrize("mixed_alignment", [False, True])
def test_grouped_kv_with_different_shapes_and_dtypes(mixed_alignment: bool) -> None:
    expected = [
        torch.arange(2 * 24 * 3, dtype=torch.float32).reshape(1, 2, 24, 3),
        torch.arange(24 * 2, dtype=torch.float16).reshape(1, 1, 24, 2),
    ]
    if mixed_alignment:
        expected = [
            torch.arange(24 * 3, dtype=torch.float16).reshape(1, 1, 24, 3),
            torch.arange(24 * 2, dtype=torch.float32).reshape(1, 1, 24, 2),
        ]
    destination = [torch.zeros_like(tensor) for tensor in expected]
    source = make_object(expected, grouped=True)
    buffer = RetrievalBuffer.get(torch.device("cpu"), 128 if mixed_alignment else 256)

    def write(objects: list[MemoryObj], starts: list[int], ends: list[int]) -> None:
        for obj, start, end in zip(objects, starts, ends, strict=True):
            for index, target in enumerate(destination):
                tensor = obj.get_tensor(index)
                assert tensor is not None
                target[:, :, start:end].copy_(tensor)

    result = make_transfer(buffer).retrieve(iter([(source, 0, 24)]), 24, write)
    assert result.all()
    for actual, wanted in zip(destination, expected, strict=True):
        torch.testing.assert_close(actual, wanted)
    assert source.metadata.shape == expected[0].shape
    assert source.metadata.ref_count == 1


def test_deferred_writes_finish_before_buffer_reuse() -> None:
    source = make_object([torch.arange(80, dtype=torch.float32).reshape(1, 1, 20, 4)])
    assert source.tensor is not None
    destination = torch.zeros_like(source.tensor)
    pending: list[tuple[MemoryObj, int, int]] = []

    def synchronize() -> None:
        while pending:
            obj, start, end = pending.pop(0)
            assert obj.tensor is not None
            destination[:, :, start:end].copy_(obj.tensor)

    def write(objects: list[MemoryObj], starts: list[int], ends: list[int]) -> None:
        pending.extend(zip(objects, starts, ends, strict=True))

    buffer = RetrievalBuffer.get(torch.device("cpu"), 128)
    result = make_transfer(buffer, synchronize=synchronize).retrieve(
        iter([(source, 0, 20)]), 20, write
    )
    assert result.all()
    assert not pending
    torch.testing.assert_close(destination, source.tensor)


def test_failed_write_drains_pending_work_and_does_not_publish() -> None:
    source = make_object([torch.ones(1, 1, 20, 4)])
    buffer = RetrievalBuffer.get(torch.device("cpu"), 128)
    pending: list[MemoryObj] = []
    completed: list[torch.Tensor] = []
    calls = 0

    def synchronize() -> None:
        while pending:
            obj = pending.pop()
            assert obj.tensor is not None
            completed.append(obj.tensor.clone())

    def write(objects: list[MemoryObj], starts: list[int], ends: list[int]) -> None:
        nonlocal calls
        calls += 1
        pending.extend(objects)
        if calls == 2:
            raise torch.OutOfMemoryError("injected destination failure")

    transfer = make_transfer(buffer, synchronize=synchronize)
    result = transfer.retrieve(iter([(source, 0, 20)]), 20, write)
    assert not result.any()
    assert calls == 2
    assert not pending
    assert all((tensor == 1).all() for tensor in completed)
    # The next request can use the same allocation after a recoverable failure.
    assert transfer.retrieve(iter([(source, 0, 20)]), 20, write).all()


@pytest.mark.parametrize("failure", ["source", "format", "token", "gap"])
def test_invalid_source_returns_no_committed_tokens(failure: str) -> None:
    source = make_object([torch.ones(1, 1, 12, 4)])
    buffer = RetrievalBuffer.get(torch.device("cpu"), 128)
    if failure == "format":
        source.metadata.fmt = MemoryFormat.BINARY
    if failure == "token":
        source = make_object([torch.ones(1, 1, 1, 100)])

    def chunks() -> Generator[RetrievalChunk, None, None]:
        if failure == "token":
            yield source, 0, 1
            return
        yield source, 0, 12
        if failure == "source":
            raise OSError("injected read failure")
        if failure == "gap":
            yield source, 13, 25

    with closing(chunks()) as iterator:
        result = make_transfer(buffer).retrieve(iterator, 25, lambda *args: None)
    assert not result.any()


def test_failed_device_fence_propagates() -> None:
    buffer = RetrievalBuffer.get(torch.device("cpu"), 128)

    def synchronize() -> None:
        raise RuntimeError("injected device failure")

    with pytest.raises(RuntimeError, match="synchronization failed"):
        make_transfer(buffer, synchronize=synchronize).retrieve(
            iter(()), 0, lambda *args: None
        )
    # A poisoned allocation cannot be handed to another engine/request.
    with buffer.lease() as unavailable:
        assert unavailable is None


def test_buffer_reservation_is_shared_and_serializes_concurrent_requests() -> None:
    first = RetrievalBuffer.get(torch.device("cpu"), 128)
    second = RetrievalBuffer.get(torch.device("cpu"), 128)
    assert first is second
    entered = threading.Event()
    released = threading.Event()
    seen: list[int] = []

    def borrower(buffer: RetrievalBuffer, owner: bool) -> None:
        with buffer.lease() as tensor:
            assert tensor is not None
            seen.append(tensor.data_ptr())
            if owner:
                tensor.fill_(17)
                entered.set()
                assert released.wait(5)
            assert (tensor == 17).all()

    with ThreadPoolExecutor(2) as executor:
        owner = executor.submit(borrower, first, True)
        assert entered.wait(5)
        waiter = executor.submit(borrower, second, False)
        assert not waiter.done()
        released.set()
        owner.result(timeout=5)
        waiter.result(timeout=5)
    assert len(set(seen)) == 1
    with pytest.raises(ValueError, match="share its budget"):
        RetrievalBuffer.get(torch.device("cpu"), 256)


def _distributed_worker(rank: int, rendezvous: str, output: str, failure: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        buffer = RetrievalBuffer.get(torch.device("cpu"), 128)
        source = make_object(
            [torch.arange(160, dtype=torch.float32).reshape(1, 2, 20, 4)]
        )
        assert source.tensor is not None
        destination = torch.full_like(source.tensor, -1)
        writes = 0
        payload_sizes = []

        def broadcast_object(obj: Any, src: int) -> Any:
            objects = [obj]
            dist.broadcast_object_list(objects, src=src)
            return objects[0]

        def agree(ready: bool) -> bool:
            flag = torch.tensor(int(ready))
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            return bool(flag.item())

        def broadcast(tensor: torch.Tensor, src: int) -> None:
            payload_sizes.append(tensor.numel())
            dist.broadcast(tensor, src=src)

        def write(objects: list[MemoryObj], starts: list[int], ends: list[int]) -> None:
            nonlocal writes
            writes += 1
            if failure == "write" and rank == 1 and writes == 2:
                raise torch.OutOfMemoryError("injected receiver OOM")
            for obj, start, end in zip(objects, starts, ends, strict=True):
                assert obj.tensor is not None
                destination[:, :, start:end].copy_(obj.tensor)

        def chunks() -> Generator[RetrievalChunk, None, None]:
            yield source, 0, 20
            if failure == "read":
                raise OSError("injected source failure")

        transfer = BroadcastTransfer(
            None if failure == "admission" and rank == 1 else buffer,
            is_source=rank == 0,
            source_rank=0,
            broadcast=broadcast,
            broadcast_object=broadcast_object,
            agree=agree,
            synchronize=lambda: None,
        )
        with closing(chunks()) as iterator:
            result = transfer.retrieve(
                iterator,
                20,
                write,
                request_id=str(rank) if failure == "identity" else "same-request",
            )
        assert result.all() if failure == "none" else not result.any()
        if failure in ("admission", "identity"):
            assert not payload_sizes
        else:
            assert payload_sizes and max(payload_sizes) <= 128

        # Every failure case must leave the collective sequence usable.
        transfer = BroadcastTransfer(
            buffer,
            is_source=rank == 0,
            source_rank=0,
            broadcast=broadcast,
            broadcast_object=broadcast_object,
            agree=agree,
            synchronize=lambda: None,
        )
        failure = "none"
        result = transfer.retrieve(iter([(source, 0, 20)]), 20, write)
        assert result.all()
        torch.testing.assert_close(destination, source.tensor)
        Path(output, f"{rank}.json").write_text(json.dumps({"ok": True}))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("failure", ["none", "admission", "write", "read", "identity"])
def test_distributed_failure_and_next_request(tmp_path: Path, failure: str) -> None:
    context = multiprocessing.get_context("spawn")
    rendezvous = (tmp_path / "rendezvous").as_uri()
    workers = [
        context.Process(
            target=_distributed_worker,
            args=(rank, rendezvous, str(tmp_path), failure),
        )
        for rank in range(2)
    ]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=30)
        assert all(worker.exitcode == 0 for worker in workers)
        for rank in range(2):
            assert json.loads((tmp_path / f"{rank}.json").read_text()) == {"ok": True}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=5)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("num_chunks", [2, 100])
def test_cuda_staging_does_not_grow_with_hit_length(num_chunks: int) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    budget = 64 * 1024
    chunk_tokens = 256
    num_tokens = num_chunks * chunk_tokens
    expected = torch.arange(2 * num_tokens * 8, dtype=torch.float32).reshape(
        1, 2, num_tokens, 8
    )
    destination = torch.empty_like(expected, device=device)
    buffer = RetrievalBuffer.get(device, budget)
    pointers: set[int] = set()

    def chunks() -> Generator[RetrievalChunk, None, None]:
        for start in range(0, num_tokens, chunk_tokens):
            yield (
                make_object([expected[:, :, start : start + chunk_tokens]]),
                start,
                (start + chunk_tokens),
            )

    def broadcast(tensor: torch.Tensor, src: int) -> None:
        assert tensor.numel() <= budget
        pointers.add(tensor.untyped_storage().data_ptr())

    def write(objects: list[MemoryObj], starts: list[int], ends: list[int]) -> None:
        for obj, start, end in zip(objects, starts, ends, strict=True):
            assert obj.tensor is not None
            destination[:, :, start:end].copy_(obj.tensor)

    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    transfer = make_transfer(
        buffer, synchronize=lambda: torch.cuda.synchronize(device), broadcast=broadcast
    )
    result = transfer.retrieve(chunks(), num_tokens, write)
    assert result.all()
    assert len(pointers) == 1
    assert torch.cuda.max_memory_allocated(device) - baseline <= budget
    torch.testing.assert_close(destination.cpu(), expected)
