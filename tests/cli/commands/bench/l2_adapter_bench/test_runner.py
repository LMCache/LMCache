# SPDX-License-Identifier: Apache-2.0
"""Check round ownership of asynchronous L2 benchmark completions."""

# Standard
from unittest.mock import Mock

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench.data import make_object_keys
from lmcache.cli.commands.bench.l2_adapter_bench.runner import bench_store
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.memory_management import MemoryObj


@pytest.mark.parametrize(
    ("completion_batches", "second_round_finishes"),
    [
        pytest.param([[0, 1], [2, 3]], True, id="late-before-current"),
        pytest.param([[0, 2], [1, 3]], True, id="late-mixed-with-current"),
        pytest.param([[0, 1]], False, id="current-round-still-times-out"),
    ],
)
def test_store_round_waits_for_its_own_tasks(
    monkeypatch: pytest.MonkeyPatch,
    completion_batches: list[list[int]],
    second_round_finishes: bool,
) -> None:
    """Late completions cannot finish or contribute success to another round."""
    adapter = Mock()
    adapter.submit_store_task.side_effect = range(4)
    adapter.pop_completed_store_tasks.side_effect = [
        {task_id: L2StoreResult(True, 1024) for task_id in batch}
        for batch in completion_batches
    ]
    # The first round times out; its tasks complete while round two is waiting.
    notifications = [False] + [True] * len(completion_batches)
    if not second_round_finishes:
        notifications.append(False)
    monkeypatch.setattr(
        "lmcache.cli.commands.bench.l2_adapter_bench.runner.wait_eventfd",
        Mock(side_effect=notifications),
    )
    keys = make_object_keys(2)
    objects = [Mock(spec=MemoryObj), Mock(spec=MemoryObj)]

    result = bench_store(
        adapter,
        in_flight=2,
        num_keys=1,
        data_size=1024,
        rounds=2,
        keys_for_round=lambda _: [[key] for key in keys],
        objs_for_round=lambda _: [[obj] for obj in objects],
        log=lambda _: None,
    )

    assert result.round_timed_out == [True, not second_round_finishes]
    assert result.success_counts == [0, 2 if second_round_finishes else 0]
    assert len(result.round_durations) == int(second_round_finishes)
