# SPDX-License-Identifier: Apache-2.0
"""Exercise round-trip verification through the public benchmark command."""

# Standard
from itertools import count
from pathlib import Path
from unittest.mock import Mock
import argparse

# Third Party
import pytest
import torch

# First Party
from lmcache.cli.commands.bench.l2_adapter_bench import L2AdapterBenchCommand
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.memory_management import MemoryObj


@pytest.mark.parametrize("skipped_key", [None, 0, 256])
@pytest.mark.no_shared_allocator
def test_verification_rejects_unwritten_load_buffers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    skipped_key: int | None,
) -> None:
    """Detect no-op loads after a successful warmup, including byte-pattern wrap."""
    num_keys = 257 if skipped_key == 256 else 1
    in_flight = 2
    stored: dict[ObjectKey, torch.Tensor] = {}
    store_ids = count()
    load_ids = count()
    completed: dict[int, L2StoreResult] = {}

    def store(keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        """Save the submitted bytes and report a completed store."""
        task_id = next(store_ids)
        for key, obj in zip(keys, objects, strict=True):
            stored[key] = obj.raw_data.clone()
        completed[task_id] = L2StoreResult(True, num_keys * 1024)
        return task_id

    def pop_stores() -> dict[int, L2StoreResult]:
        """Consume completed store results."""
        result = completed.copy()
        completed.clear()
        return result

    def load(keys: list[ObjectKey], objects: list[MemoryObj]) -> int:
        """Copy warmup data but leave the selected measured buffer untouched."""
        task_id = next(load_ids)
        for index, (key, obj) in enumerate(zip(keys, objects, strict=True)):
            if task_id < in_flight or index != skipped_key:
                obj.raw_data.copy_(stored[key])
        return task_id

    adapter = Mock()
    adapter.submit_store_task.side_effect = store
    adapter.pop_completed_store_tasks.side_effect = pop_stores
    adapter.submit_lookup_and_lock_task.side_effect = count()
    adapter.query_lookup_and_lock_result.return_value = Mock(
        popcount=Mock(return_value=0)
    )
    adapter.submit_load_task.side_effect = load
    adapter.query_load_result.return_value = Mock(popcount=Mock(return_value=num_keys))
    monkeypatch.setattr(
        "lmcache.v1.distributed.l2_adapters.create_l2_adapter",
        Mock(return_value=adapter),
    )
    monkeypatch.setattr(
        "lmcache.cli.commands.bench.l2_adapter_bench.runner.wait_eventfd",
        Mock(return_value=True),
    )

    command = L2AdapterBenchCommand()
    parser = argparse.ArgumentParser()
    command.register(parser.add_subparsers())
    args = parser.parse_args(
        [
            "l2",
            "--l2-adapter",
            '{"type":"mock","max_size_gb":0.01,"mock_bandwidth_gb":1}',
            "--num-keys",
            str(num_keys),
            "--in-flight",
            str(in_flight),
            "--data-size-kb",
            "1",
            "--rounds",
            "1",
            "--warmup-rounds",
            "1",
            "--no-skip-verify",
            "--output",
            str(tmp_path / "result.json"),
            "--format",
            "json",
        ]
    )

    if skipped_key is None:
        command.execute(args)
        assert "data verified OK" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit) as exc:
            command.execute(args)
        assert exc.value.code == 1
        assert f"2/{num_keys * in_flight} keys have data mismatches" in (
            capsys.readouterr().out
        )
    adapter.close.assert_called_once_with()
