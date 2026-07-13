# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression tests for EngineDrivenTransferContext.

The single-group ("backward compatible") store/retrieve paths of
``EngineDrivenTransferContext`` previously had no coverage at all,
which let two breakages slip through unnoticed:

- ``submit_store`` dropped the required positional ``blocks_per_chunk``
  argument of ``gather_paged_kv_to_cpu`` (TypeError on every
  single-group store).
- ``submit_retrieve`` lost its ``COMMIT_RETRIEVE`` call, leaking the
  server-side pending SHM read locks on every retrieve.

These tests exercise the worker-side control flow with a fake
``EngineDrivenContext`` and gather/scatter stubs that mirror the real
function signatures (so a missing positional argument fails loudly).
No CUDA, no MP server, no lmc_ops kernels required.
"""

# Standard
from typing import Any

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.engine import PrepareRetrieveResponse
from lmcache.v1.multiprocess.transfer_context import worker_transfer as wt
from lmcache.v1.multiprocess.transfer_context.base import (
    PinnedBufferPool,
    _serialize_single_group_chunks,
)
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)


class FakeEngineDrivenContext:
    """Records every call; mimics the pickle-mode EngineDrivenContext."""

    def __init__(self) -> None:
        self.mq_timeout = 5.0
        self.calls: list[tuple[str, tuple]] = []
        # Configurable multi-group retrieve responses (group_idx -> chunks
        # or None for a miss).
        self.retrieve_groups: dict[int, list[torch.Tensor] | None] = {}
        self.fail_group_commit = False

    # ── store ────────────────────────────────────────────────────────
    def prepare_store(self, key: Any, instance_id: int):
        self.calls.append(("prepare_store", (key, instance_id)))
        return None  # pickle mode: no pre-allocated buffers

    def commit_store(self, key: Any, instance_id: int, chunks) -> bool:
        self.calls.append(("commit_store", (key, instance_id, len(chunks))))
        return True

    def commit_store_group_raw_async(
        self, key: Any, instance_id: int, group_idx: int, cpu_data: bytes
    ) -> MessagingFuture:
        self.calls.append(
            ("commit_store_group_raw_async", (key, instance_id, group_idx))
        )
        if self.fail_group_commit:
            raise RuntimeError("injected commit failure")
        fut: MessagingFuture = MessagingFuture()
        fut.set_result(True)
        return fut

    def commit_store_group_delta_raw_async(
        self,
        key: Any,
        instance_id: int,
        group_idx: int,
        skip_count: int,
        cpu_data: bytes,
    ) -> MessagingFuture:
        self.calls.append(
            (
                "commit_store_group_delta_raw_async",
                (key, instance_id, group_idx, skip_count),
            )
        )
        fut: MessagingFuture = MessagingFuture()
        fut.set_result(True)
        return fut

    # ── retrieve ─────────────────────────────────────────────────────
    def prepare_retrieve(self, key: Any, instance_id: int):
        self.calls.append(("prepare_retrieve", (key, instance_id)))
        return [torch.zeros(2, 1, 16, 8)]

    def prepare_retrieve_group_raw_async(
        self, key: Any, instance_id: int, group_idx: int
    ) -> MessagingFuture:
        self.calls.append(
            ("prepare_retrieve_group_raw_async", (key, instance_id, group_idx))
        )
        fut: MessagingFuture = MessagingFuture()
        chunks = self.retrieve_groups.get(group_idx)
        if chunks is None:
            fut.set_result(PrepareRetrieveResponse(success=False, data=b""))
        else:
            fut.set_result(
                PrepareRetrieveResponse(
                    success=True, data=_serialize_single_group_chunks(chunks)
                )
            )
        return fut

    def commit_retrieve(self, key: Any, instance_id: int) -> bool:
        self.calls.append(("commit_retrieve", (key, instance_id)))
        return True

    def close(self) -> None:
        self.calls.append(("close", ()))

    # helper
    def count(self, name: str) -> int:
        return sum(1 for c, _ in self.calls if c == name)


def _make_ctx(
    num_groups: int = 1, tokens_per_chunk: int = 32
) -> tuple[EngineDrivenTransferContext, FakeEngineDrivenContext]:
    ctx = EngineDrivenTransferContext()
    fake = FakeEngineDrivenContext()
    ctx._engine_driven_context = fake  # type: ignore[assignment]
    ctx._engine_group_infos = [
        EngineGroupInfo(
            engine_group_id=0, layer_indices=tuple(range(g, g + 1)), tokens_per_block=16
        )
        for g in range(num_groups)
    ]
    ctx._lmcache_tokens_per_chunk = tokens_per_chunk
    ctx._layout_hints = None
    ctx._engine_kv_format = None
    return ctx, fake


# ─── Signature-faithful gather/scatter stubs ─────────────────────────────────
# Positional parameters mirror the real functions in base.py exactly, so a
# call site that drops a required positional argument raises TypeError here
# just like it would in production.


def _make_gather_stub(record: dict):
    def fake_gather_paged_kv_to_cpu(
        kv_caches,
        block_ids,
        blocks_per_chunk,
        layout_hints=None,
        engine_kv_format=None,
        out=None,
        chunk_indices=None,
        pinned_pool=None,
    ):
        record["blocks_per_chunk"] = blocks_per_chunk
        record["block_ids"] = list(block_ids)
        record["out"] = out
        return [torch.zeros(2, 1, 16, 8)]

    return fake_gather_paged_kv_to_cpu


def _make_scatter_stub(record: dict):
    def fake_scatter_cpu_to_paged_kv(
        kv_caches,
        block_ids,
        chunks,
        blocks_per_chunk,
        skip_first_n_tokens=0,
        layout_hints=None,
        engine_kv_format=None,
    ):
        record["blocks_per_chunk"] = blocks_per_chunk
        record["num_chunks"] = len(chunks)

    return fake_scatter_cpu_to_paged_kv


# ─── Test 1: single-group store passes blocks_per_chunk (regression B2) ─────


def test_single_group_store_passes_blocks_per_chunk(monkeypatch):
    ctx, fake = _make_ctx(num_groups=1)
    record: dict = {}
    monkeypatch.setattr(wt, "gather_paged_kv_to_cpu", _make_gather_stub(record))

    future = ctx.submit_store(
        "req-1",
        key="key-1",
        instance_id=7,
        kv_caches={"0": torch.zeros(2, 4, 16, 8)},
        block_ids=[[0, 1]],
        _event=None,
        blocks_in_chunk=2,
    )
    assert future.result(timeout=1.0) is True
    # The required positional argument must arrive at the gather.
    assert record["blocks_per_chunk"] == 2
    assert record["block_ids"] == [0, 1]
    assert fake.count("prepare_store") == 1
    assert fake.count("commit_store") == 1


# ─── Test 2: single-group retrieve calls COMMIT_RETRIEVE (regression B3) ─────


def test_single_group_retrieve_calls_commit_retrieve(monkeypatch):
    ctx, fake = _make_ctx(num_groups=1)
    record: dict = {}
    monkeypatch.setattr(wt, "scatter_cpu_to_paged_kv", _make_scatter_stub(record))

    future = ctx.submit_retrieve(
        "req-2",
        key="key-2",
        instance_id=7,
        kv_caches={"0": torch.zeros(2, 4, 16, 8)},
        block_ids=[[0, 1]],
        _event=None,
        blocks_in_chunk=2,
    )
    assert future.result(timeout=1.0) is True
    assert record["blocks_per_chunk"] == 2
    # COMMIT_RETRIEVE releases the server-side pending SHM read locks;
    # it must be sent exactly once per retrieve.
    assert fake.count("commit_retrieve") == 1


def test_single_group_retrieve_scatter_failure_still_commits(monkeypatch):
    ctx, fake = _make_ctx(num_groups=1)

    def raising_scatter(*args, **kwargs):
        raise RuntimeError("scatter failed")

    monkeypatch.setattr(wt, "scatter_cpu_to_paged_kv", raising_scatter)
    future = ctx.submit_retrieve(
        "req-3",
        key="key-3",
        instance_id=7,
        kv_caches={"0": torch.zeros(2, 4, 16, 8)},
        block_ids=[[0, 1]],
        _event=None,
        blocks_in_chunk=2,
    )
    assert future.result(timeout=1.0) is False
    assert fake.count("commit_retrieve") == 1


# ─── Test 3: multi-group retrieve uses per-group requests ────────────────────


def test_multi_group_retrieve_per_group_roundtrip(monkeypatch):
    ctx, fake = _make_ctx(num_groups=2)
    fake.retrieve_groups = {
        0: [torch.randn(2, 1, 16, 8)],
        1: [torch.randn(1, 1, 32, 4)],
    }
    record: dict = {}

    def fake_scatter_multi(
        kv_caches,
        block_ids,
        group_chunks,
        engine_group_infos,
        lmcache_tokens_per_chunk,
        skip_first_n_tokens=0,
        layout_hints=None,
    ):
        record["num_groups"] = len(group_chunks)
        record["shapes"] = [tuple(g[0].shape) for g in group_chunks]

    monkeypatch.setattr(wt, "scatter_cpu_multi_group_to_paged_kv", fake_scatter_multi)

    future = ctx.submit_retrieve(
        "req-4",
        key="key-4",
        instance_id=7,
        kv_caches={"0": torch.zeros(2, 4, 16, 8), "1": torch.zeros(2, 4, 32, 4)},
        block_ids=[[0, 1], [0, 1]],
        _event=None,
        blocks_in_chunk=2,
    )
    assert future.result(timeout=1.0) is True
    # One PREPARE_RETRIEVE_GROUP per group, in order.
    assert fake.count("prepare_retrieve_group_raw_async") == 2
    assert record["num_groups"] == 2
    assert record["shapes"] == [(2, 1, 16, 8), (1, 1, 32, 4)]
    assert fake.count("commit_retrieve") == 1


def test_multi_group_retrieve_partial_miss_is_overall_miss(monkeypatch):
    ctx, fake = _make_ctx(num_groups=2)
    fake.retrieve_groups = {0: [torch.randn(2, 1, 16, 8)], 1: None}  # group 1 misses
    scatter_called: list = []
    monkeypatch.setattr(
        wt,
        "scatter_cpu_multi_group_to_paged_kv",
        lambda *a, **k: scatter_called.append(1),
    )

    future = ctx.submit_retrieve(
        "req-5",
        key="key-5",
        instance_id=7,
        kv_caches={"0": torch.zeros(2, 4, 16, 8), "1": torch.zeros(2, 4, 32, 4)},
        block_ids=[[0, 1], [0, 1]],
        _event=None,
        blocks_in_chunk=2,
    )
    assert future.result(timeout=1.0) is False
    assert not scatter_called  # all-or-nothing: no partial scatter
    assert fake.count("commit_retrieve") == 1


# ─── Test 4: multi-group store releases pinned buffers on failure ────────────


def test_multi_group_store_releases_buffers_on_commit_failure(monkeypatch):
    ctx, fake = _make_ctx(num_groups=2)
    fake.fail_group_commit = True
    chunks_g0 = [torch.zeros(64, dtype=torch.uint8)]
    chunks_g1 = [torch.zeros(128, dtype=torch.uint8)]
    monkeypatch.setattr(
        wt,
        "gather_paged_kv_multi_group_to_cpu",
        lambda *a, **k: [chunks_g0, chunks_g1],
    )

    with pytest.raises(RuntimeError, match="injected commit failure"):
        ctx.submit_store(
            "req-6",
            key="key-6",
            instance_id=7,
            kv_caches={"0": torch.zeros(2, 4, 16, 8), "1": torch.zeros(2, 4, 32, 4)},
            block_ids=[[0, 1], [0, 1]],
            _event=None,
            blocks_in_chunk=2,
        )
    # Buffers must be back in the pool despite the exception (finally).
    stats = ctx._pinned_pool.stats()
    assert stats["idle_bytes"] == 64 + 128, stats
    ctx.close()


def test_multi_group_store_success_uses_group_commits(monkeypatch):
    ctx, fake = _make_ctx(num_groups=2)
    monkeypatch.setattr(
        wt,
        "gather_paged_kv_multi_group_to_cpu",
        lambda *a, **k: [
            [torch.zeros(64, dtype=torch.uint8)],
            [torch.zeros(128, dtype=torch.uint8)],
        ],
    )
    monkeypatch.delenv("LMCACHE_MP_DELTA_STORE", raising=False)
    future = ctx.submit_store(
        "req-7",
        key="key-7",
        instance_id=7,
        kv_caches={"0": torch.zeros(2, 4, 16, 8), "1": torch.zeros(2, 4, 32, 4)},
        block_ids=[[0, 1], [0, 1]],
        _event=None,
        blocks_in_chunk=2,
    )
    assert future.result(timeout=1.0) is True
    # Delta-store defaults to OFF (skip_count is hardwired to 0), so the
    # plain per-group commit must be used.
    assert fake.count("commit_store_group_raw_async") == 2
    assert fake.count("commit_store_group_delta_raw_async") == 0
    ctx.close()


# ─── Test 5: PinnedBufferPool capacity cap / LRU trim ────────────────────────


def test_pinned_pool_trims_to_capacity():
    pool = PinnedBufferPool(capacity_bytes=1000)
    # Inject plain CPU tensors via release() -- no pinning, no CUDA needed.
    pool.release([torch.zeros(100, dtype=torch.uint8) for _ in range(20)])  # 2000 B
    stats = pool.stats()
    assert stats["idle_bytes"] <= 1000
    # Re-acquire pops from the pool without allocating.
    got = pool.acquire((100,), torch.uint8, count=2)
    assert len(got) == 2
    assert pool.stats()["idle_bytes"] == stats["idle_bytes"] - 200


def test_pinned_pool_unlimited_when_cap_disabled():
    pool = PinnedBufferPool(capacity_bytes=0)
    pool.release([torch.zeros(1000, dtype=torch.uint8) for _ in range(5)])
    assert pool.stats()["idle_bytes"] == 5000
