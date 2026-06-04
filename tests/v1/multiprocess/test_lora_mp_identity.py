# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    LMCacheMPSchedulerAdapter,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
    ParallelStrategy,
)


def _parallel_strategy() -> ParallelStrategy:
    return ParallelStrategy(
        use_mla=False,
        kv_world_size=2,
        kv_worker_id=1,
        actual_world_size=2,
        actual_worker_id=1,
        tp_size=2,
        pp_size=1,
    )


def test_scheduler_create_key_includes_lora_name() -> None:
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter.model_name = "model"
    adapter.parallel_strategy = _parallel_strategy()

    key = adapter._create_key(
        [1, 2, 3],
        0,
        3,
        "req",
        cache_salt="tenant",
        lora_name="adapter_a",
    )

    assert key.worker_id is None
    assert key.cache_salt == "tenant"
    assert key.lora_name == "adapter_a"


def test_worker_create_key_includes_lora_name() -> None:
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.model_name = "model"
    adapter.parallel_strategy = _parallel_strategy()

    key = adapter._create_key(
        [1, 2, 3],
        0,
        3,
        "req",
        cache_salt="tenant",
        lora_name="adapter_a",
    )

    assert key.worker_id == 1
    assert key.cache_salt == "tenant"
    assert key.lora_name == "adapter_a"


def test_batched_store_requires_lora_names_length_to_match(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    calls = []

    def fake_submit_store_request(
        request_id: str,
        op: LoadStoreOp,
        event: object,
        cache_salt: str = "",
        lora_name: str = "",
    ) -> None:
        calls.append((request_id, cache_salt, lora_name))

    monkeypatch.setattr(adapter, "submit_store_request", fake_submit_store_request)

    op = LoadStoreOp(token_ids=[1, 2, 3], block_ids=[[1]], start=0, end=3)
    adapter.batched_submit_store_requests(
        ["req-a", "req-b"],
        [op, op],
        event=object(),
        cache_salts=["tenant-a", "tenant-b"],
        lora_names=["adapter_a", "adapter_b"],
    )

    assert calls == [
        ("req-a", "tenant-a", "adapter_a"),
        ("req-b", "tenant-b", "adapter_b"),
    ]


def test_batched_store_rejects_lora_names_length_mismatch() -> None:
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    op = LoadStoreOp(token_ids=[1, 2, 3], block_ids=[[1]], start=0, end=3)

    with pytest.raises(ValueError, match="lora_names length"):
        adapter.batched_submit_store_requests(
            ["req-a", "req-b"],
            [op, op],
            event=object(),
            lora_names=["adapter_a"],
        )


def test_batched_retrieve_requires_lora_names_length_to_match(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    calls = []

    def fake_submit_retrieve_request(
        request_id: str,
        op: LoadStoreOp,
        event: object,
        cache_salt: str = "",
        lora_name: str = "",
    ) -> None:
        calls.append((request_id, cache_salt, lora_name))

    monkeypatch.setattr(
        adapter, "submit_retrieve_request", fake_submit_retrieve_request
    )

    op = LoadStoreOp(token_ids=[1, 2, 3], block_ids=[[1]], start=0, end=3)
    adapter.batched_submit_retrieve_requests(
        ["req-a", "req-b"],
        [op, op],
        event=object(),
        cache_salts=["tenant-a", "tenant-b"],
        lora_names=["adapter_a", "adapter_b"],
    )

    assert calls == [
        ("req-a", "tenant-a", "adapter_a"),
        ("req-b", "tenant-b", "adapter_b"),
    ]


def test_batched_retrieve_rejects_lora_names_length_mismatch() -> None:
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    op = LoadStoreOp(token_ids=[1, 2, 3], block_ids=[[1]], start=0, end=3)

    with pytest.raises(ValueError, match="lora_names length"):
        adapter.batched_submit_retrieve_requests(
            ["req-a", "req-b"],
            [op, op],
            event=object(),
            lora_names=["adapter_a"],
        )
