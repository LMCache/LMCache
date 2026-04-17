# SPDX-License-Identifier: Apache-2.0

"""Tests for ``lmcache bench trace-replay``.

End-to-end: records a tiny scripted trace, then invokes the bench
sub-target's ``run()`` against a fresh StorageManager and verifies
the CSV/JSON exports and terminal summary behave as advertised.
"""

# Future
from __future__ import annotations

# Standard
import argparse
import json
import os
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.cli.commands.bench import BenchCommand
from lmcache.cli.commands.bench import trace_replay as cmd_module
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace.decorator import set_tracing_enabled
from lmcache.v1.mp_observability.trace.recorder import StorageTraceRecorder
import lmcache.v1.mp_observability.event_bus as _bus_module


def _should_use_lazy() -> bool:
    return torch.cuda.is_available()


def _make_sm_config() -> StorageManagerConfig:
    memory = L1MemoryManagerConfig(
        size_in_bytes=64 * 1024 * 1024,
        use_lazy=_should_use_lazy(),
        init_size_in_bytes=32 * 1024 * 1024,
        align_bytes=0x1000,
    )
    l1 = L1ManagerConfig(memory_config=memory)
    return StorageManagerConfig(
        l1_manager_config=l1,
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


@pytest.fixture(autouse=True)
def restore_global_bus():
    saved = _bus_module._global_bus
    yield
    _bus_module._global_bus = saved
    set_tracing_enabled(False)


@pytest.fixture
def recorded_trace(tmp_path):
    """Record reserve_write + finish_write into a trace file."""
    path = str(tmp_path / "replay.lct")
    sm_config = _make_sm_config()
    layout = MemoryLayoutDesc(shapes=[torch.Size([16, 16])], dtypes=[torch.float16])
    keys = [ObjectKey(chunk_hash=b"\x01", model_name="t", kv_rank=0)]

    bus = EventBus(EventBusConfig(enabled=True))
    _bus_module._global_bus = bus
    bus.start()
    sm = StorageManager(sm_config)
    rec = StorageTraceRecorder(path)
    rec.attach_storage_config(sm_config)
    bus.register_subscriber(rec)
    try:
        sm.reserve_write(keys, layout, mode="new")
        sm.finish_write(keys)
        time.sleep(0.2)
        bus._drain_all()
    finally:
        bus.stop()
        sm.close()
    return path


class TestArgumentWiring:
    def test_trace_replay_registered_as_bench_target(self):
        """``bench trace-replay <file> ...`` is parseable via the
        full BenchCommand.register() chain."""
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="command")
        BenchCommand().register(subs)
        args = parser.parse_args(
            [
                "bench",
                "trace-replay",
                "/tmp/x.lct",
                "--l1-size-gb",
                "0.0625",
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.bench_target == "trace-replay"
        assert args.trace_path == "/tmp/x.lct"
        assert args.pace == "asap"

    def test_pace_choices_enforced(self):
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers(dest="command")
        BenchCommand().register(subs)
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "bench",
                    "trace-replay",
                    "/tmp/x.lct",
                    "--l1-size-gb",
                    "0.0625",
                    "--eviction-policy",
                    "LRU",
                    "--pace",
                    "bogus",
                ]
            )


class TestRun:
    def test_run_writes_csv_and_json(self, recorded_trace, tmp_path):
        output_dir = str(tmp_path / "out")
        args = argparse.Namespace(
            trace_path=recorded_trace,
            pace="asap",
            output_dir=output_dir,
            no_csv=False,
            json=True,
            quiet=True,
            # StorageManagerConfig fields — mirror _make_sm_config.
            l1_size_gb=0.0625,
            l1_use_lazy=_should_use_lazy(),
            l1_init_size_gb=0.03125,
            l1_align_bytes=0x1000,
            l1_write_ttl_seconds=600,
            l1_read_ttl_seconds=300,
            eviction_policy="LRU",
            eviction_trigger_watermark=0.8,
            eviction_ratio=0.2,
            l2_store_policy="default",
            l2_prefetch_policy="default",
            l2_prefetch_max_in_flight=8,
            l2_adapter=[],
        )
        cmd_module.run(args)

        csv_path = os.path.join(output_dir, "trace_replay_ops.csv")
        json_path = os.path.join(output_dir, "trace_replay_summary.json")
        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)

        with open(json_path) as f:
            data = json.load(f)
        assert "ops" in data
        # Both recorded ops should appear in the summary.
        qns = list(data["ops"])
        assert any("reserve_write" in qn for qn in qns)
        assert any("finish_write" in qn for qn in qns)

    def test_quiet_suppresses_terminal_output(self, recorded_trace, tmp_path, capsys):
        output_dir = str(tmp_path / "out")
        args = argparse.Namespace(
            trace_path=recorded_trace,
            pace="asap",
            output_dir=output_dir,
            no_csv=True,
            json=False,
            quiet=True,
            l1_size_gb=0.0625,
            l1_use_lazy=_should_use_lazy(),
            l1_init_size_gb=0.03125,
            l1_align_bytes=0x1000,
            l1_write_ttl_seconds=600,
            l1_read_ttl_seconds=300,
            eviction_policy="LRU",
            eviction_trigger_watermark=0.8,
            eviction_ratio=0.2,
            l2_store_policy="default",
            l2_prefetch_policy="default",
            l2_prefetch_max_in_flight=8,
            l2_adapter=[],
        )
        cmd_module.run(args)
        captured = capsys.readouterr()
        # No "Trace Replay Result" heading when quiet.
        assert "Trace Replay Result" not in captured.out
