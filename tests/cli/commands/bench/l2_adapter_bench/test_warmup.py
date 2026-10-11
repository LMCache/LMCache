# SPDX-License-Identifier: Apache-2.0
"""Check warmup filtering through the public L2 benchmark command."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import argparse
import json
import statistics

# Third Party
import pytest

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench import L2AdapterBenchCommand
from lmcache.v1.distributed.internal_api import L2StoreResult


# Each input round is (duration in seconds, successful keys), with four keys
# split across two submits. None means the round times out.
@pytest.mark.parametrize("operation", ["store", "lookup", "load"])
@pytest.mark.parametrize(
    "rounds,warmup,expected_durations,expected_timeouts,expected_success",
    [
        pytest.param(
            [(None, 0), (0.1, 4), (0.2, 4)],
            0,
            [0.1, 0.2],
            1,
            8,
            id="no-warmup",
        ),
        pytest.param(
            [(None, 0), (0.1, 4), (0.2, 4)],
            1,
            [0.1, 0.2],
            0,
            8,
            id="warmup-timeout",
        ),
        pytest.param(
            [(0.1, 4), (None, 0), (0.2, 4)],
            1,
            [0.2],
            1,
            4,
            id="measurement-timeout",
        ),
        pytest.param(
            [(0.1, 4), (None, 2), (0.2, 4)],
            1,
            [0.2],
            1,
            6,
            id="partial-measurement-timeout",
        ),
        pytest.param(
            [(0.1, 4), (None, 0), (None, 0)],
            1,
            [],
            2,
            0,
            id="all-measurements-time-out",
        ),
        pytest.param(
            [(0.1, 4), (0.2, 4), (0.3, 4)],
            1,
            [0.2, 0.3],
            0,
            8,
            id="all-rounds-complete",
        ),
        pytest.param(
            [(None, 2), (0.1, 4), (None, 0), (0.2, 4)],
            2,
            [0.2],
            1,
            4,
            id="mixed-warmup-rounds",
        ),
        pytest.param([(None, 0)], 0, [], 1, 0, id="single-timeout"),
    ],
)
@pytest.mark.no_shared_allocator
def test_benchmark_excludes_warmup_from_reported_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    rounds: list[tuple[float | None, int]],
    warmup: int,
    expected_durations: list[float],
    expected_timeouts: int,
    expected_success: int,
) -> None:
    """Report only measurement rounds, retaining timeout and partial key counts."""
    adapter = Mock()
    task_ids = range(2 * len(rounds))
    adapter.submit_store_task.side_effect = task_ids
    adapter.submit_lookup_and_lock_task.side_effect = task_ids
    adapter.submit_load_task.side_effect = task_ids
    adapter.get_store_event_fd.return_value = 123
    adapter.get_lookup_and_lock_event_fd.return_value = 123
    adapter.get_load_event_fd.return_value = 123

    store_completions: list[dict[int, L2StoreResult]] = []
    lookup_completions: dict[int, Mock] = {}
    load_completions: dict[int, Mock] = {}
    notifications: list[bool] = []
    clock_values: list[float] = []
    for round_index, (duration, successful_keys) in enumerate(rounds):
        completed_ids = range(2 * round_index, 2 * round_index + successful_keys // 2)
        if successful_keys:
            notifications.append(True)
            store_completions.append(
                {task_id: L2StoreResult(True, 2048) for task_id in completed_ids}
            )
            for task_id in completed_ids:
                lookup_completions[task_id] = Mock(popcount=Mock(return_value=2))
                load_completions[task_id] = Mock(popcount=Mock(return_value=2))
        if duration is None:
            notifications.append(False)
        clock_values.extend([0.0, duration if duration is not None else 120.0])

    adapter.pop_completed_store_tasks.side_effect = store_completions

    def query_lookup_result(task_id: int) -> Mock | None:
        """Consume the configured lookup completion for a task."""
        return lookup_completions.pop(task_id, None)

    def query_load_result(task_id: int) -> Mock | None:
        """Consume the configured load completion for a task."""
        return load_completions.pop(task_id, None)

    adapter.query_lookup_and_lock_result.side_effect = query_lookup_result
    adapter.query_load_result.side_effect = query_load_result
    monkeypatch.setattr(
        "lmcache.v1.distributed.l2_adapters.create_l2_adapter",
        Mock(return_value=adapter),
    )
    monkeypatch.setattr(
        "lmcache.cli.commands.bench.l2_adapter_bench.runner.wait_eventfd",
        Mock(side_effect=notifications),
    )
    monkeypatch.setattr(
        "lmcache.cli.commands.bench.l2_adapter_bench.runner.time",
        SimpleNamespace(perf_counter=Mock(side_effect=clock_values)),
    )

    command = L2AdapterBenchCommand()
    parser = argparse.ArgumentParser()
    command.register(parser.add_subparsers())
    output = tmp_path / "result.json"
    args = parser.parse_args(
        [
            "l2",
            "--l2-adapter",
            '{"type":"mock","max_size_gb":0.01,"mock_bandwidth_gb":1}',
            "--only",
            operation,
            "--num-keys",
            "2",
            "--in-flight",
            "2",
            "--data-size-kb",
            "1",
            "--rounds",
            str(len(rounds) - warmup),
            "--warmup-rounds",
            str(warmup),
            "--lookup-max-hit-rate",
            "1",
            "--quiet",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    command.execute(args)

    metrics = json.loads(output.read_text())["metrics"]["op_0"]
    measurement_rounds = len(rounds) - warmup
    assert metrics["rounds"] == measurement_rounds
    assert metrics["rounds_timed_out"] == expected_timeouts
    assert metrics["total_keys"] == 4 * measurement_rounds
    assert metrics["total_success"] == expected_success
    expected_avg = statistics.mean(expected_durations) if expected_durations else 0.0
    assert metrics["duration_avg_ms"] == pytest.approx(expected_avg * 1000)
    assert metrics["duration_min_ms"] == pytest.approx(
        min(expected_durations, default=0.0) * 1000
    )
    assert metrics["duration_max_ms"] == pytest.approx(
        max(expected_durations, default=0.0) * 1000
    )
    if operation == "lookup":
        expected_rate = expected_success / (4 * measurement_rounds)
        assert metrics["actual_hit_rate"] == pytest.approx(round(expected_rate, 4))
    adapter.close.assert_called_once_with()
