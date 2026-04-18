# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``lmcache trace`` CLI command.

End-to-end: ``info`` is smoke-tested against a tiny trace file written
through the real recorder.  ``replay`` is exercised against the same
fixture via the driver (argparse wiring only — the driver itself has
its own tests in ``tests/cli/commands/trace``) plus an end-to-end
replay that exercises the CSV/JSON summary export and terminal
metrics output.  ``record`` is a stub and simply asserts its non-zero
exit code.
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
from lmcache.cli.commands.trace import TraceCommand
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace.decorator import (
    publish_call_event,
    set_tracing_enabled,
)
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


def _replay_namespace(
    trace_path: str, output_dir: str, **overrides
) -> argparse.Namespace:
    """Build a Namespace matching the ``trace replay`` parser defaults.

    Callers override the handful of fields they care about (``no_csv``,
    ``json``, ``quiet``); everything else is pinned to values that
    reproduce :func:`_make_sm_config` so the recorded trace replays
    cleanly.
    """
    defaults = dict(
        trace_target="replay",
        trace_path=trace_path,
        verbose=False,
        jsonl_out=None,
        output_dir=output_dir,
        no_csv=False,
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
        # Observability flags mirrored from ``add_observability_args``.
        # Defaults chosen to be safe for unit tests: the bus is enabled
        # so internal SM events flow, but metrics are off to avoid
        # binding a Prometheus port inside the test runner.
        disable_observability=False,
        disable_metrics=True,
        disable_logging=False,
        enable_tracing=False,
        otlp_endpoint=None,
        event_bus_queue_size=10_000,
        prometheus_port=9090,
        metrics_sample_rate=0.01,
        lookup_hash_log_dir="",
        lookup_hash_log_rotation_interval=6 * 3600,
        lookup_hash_log_rotation_max_size=100 * 1024 * 1024,
        lookup_hash_log_max_files=100,
        # ``--trace-level`` / ``--trace-output`` are registered on the
        # replay parser (shared with ``lmcache server``) but ignored at
        # runtime — ``_run_replay`` clobbers them to ``None`` before
        # building the ObservabilityConfig.  We mirror both states
        # (the parser-default ``None`` and the explicit override) by
        # pinning them here.
        trace_level=None,
        trace_output=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def restore_global_bus():
    saved = _bus_module._global_bus
    yield
    _bus_module._global_bus = saved
    set_tracing_enabled(False)


@pytest.fixture
def cmd() -> TraceCommand:
    return TraceCommand()


@pytest.fixture
def parser(cmd: TraceCommand) -> argparse.ArgumentParser:
    """Argparse root with only the ``trace`` subcommand registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")
    cmd.register(sub)
    return p


@pytest.fixture
def small_trace(tmp_path):
    """Write a tiny trace file with two records through the real stack."""
    path = str(tmp_path / "small.lct")
    bus = EventBus(EventBusConfig(enabled=True))
    _bus_module._global_bus = bus
    bus.start()
    rec = StorageTraceRecorder(path)
    bus.register_subscriber(rec)
    try:
        publish_call_event("pkg.mod.foo", {"x": 1})
        publish_call_event("pkg.mod.bar", {"y": 2})
        time.sleep(0.2)
        bus._drain_all()
    finally:
        bus.stop()
    return path


@pytest.fixture
def replayable_trace(tmp_path):
    """Record reserve_write + finish_write into a replayable trace file.

    Uses the real StorageManager stack so the trace references
    qualnames that the default dispatcher actually handles — needed
    for the end-to-end replay tests below.
    """
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


class TestMetadata:
    def test_name(self, cmd):
        assert cmd.name() == "trace"

    def test_help(self, cmd):
        assert "trace" in cmd.help().lower()


class TestArgumentParsing:
    def test_info_requires_path(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["trace", "info"])

    def test_info_accepts_path(self, parser):
        args = parser.parse_args(["trace", "info", "/tmp/x.lct"])
        assert args.trace_target == "info"
        assert args.trace_path == "/tmp/x.lct"

    def test_replay_requires_storage_flags(self, parser):
        """``add_storage_manager_args`` marks --l1-size-gb and
        --eviction-policy as required; omitting them triggers a parse
        error."""
        with pytest.raises(SystemExit):
            parser.parse_args(["trace", "replay", "/tmp/x.lct"])

    def test_replay_accepts_storage_flags(self, parser):
        args = parser.parse_args(
            [
                "trace",
                "replay",
                "/tmp/x.lct",
                "--l1-size-gb",
                "0.0625",  # 64 MB
                "--eviction-policy",
                "LRU",
            ]
        )
        assert args.trace_target == "replay"
        assert args.l1_size_gb == 0.0625
        assert args.eviction_policy == "LRU"

    def test_replay_accepts_output_flags(self, parser):
        """``--output-dir`` / ``--no-csv`` / ``--json`` / ``-q`` are
        part of ``trace replay`` after the bench merge."""
        args = parser.parse_args(
            [
                "trace",
                "replay",
                "/tmp/x.lct",
                "--l1-size-gb",
                "0.0625",
                "--eviction-policy",
                "LRU",
                "--output-dir",
                "/tmp/out",
                "--no-csv",
                "--json",
                "-q",
            ]
        )
        assert args.output_dir == "/tmp/out"
        assert args.no_csv is True
        assert args.json is True
        assert args.quiet is True


class TestInfoSubcommand:
    def test_info_prints_summary(self, cmd, small_trace, capsys):
        args = argparse.Namespace(
            trace_target="info",
            trace_path=small_trace,
        )
        cmd.execute(args)
        out = capsys.readouterr().out
        assert "Trace file:" in out
        assert "level:" in out
        assert "pkg.mod.foo" in out
        assert "pkg.mod.bar" in out


class TestReplaySubcommand:
    def test_replay_writes_csv_and_json(self, cmd, replayable_trace, tmp_path):
        """End-to-end: ``trace replay`` writes both CSV and JSON
        summaries under ``--output-dir`` and the JSON includes one
        entry per recorded qualname."""
        output_dir = str(tmp_path / "out")
        args = _replay_namespace(
            replayable_trace,
            output_dir,
            no_csv=False,
            json=True,
            quiet=True,
        )
        cmd.execute(args)

        csv_path = os.path.join(output_dir, "trace_replay_ops.csv")
        json_path = os.path.join(output_dir, "trace_replay_summary.json")
        assert os.path.exists(csv_path)
        assert os.path.exists(json_path)

        with open(json_path) as f:
            data = json.load(f)
        assert "ops" in data
        qns = list(data["ops"])
        assert any("reserve_write" in qn for qn in qns)
        assert any("finish_write" in qn for qn in qns)

    def test_replay_quiet_suppresses_terminal_output(
        self, cmd, replayable_trace, tmp_path, capsys
    ):
        output_dir = str(tmp_path / "out")
        args = _replay_namespace(
            replayable_trace,
            output_dir,
            no_csv=True,
            json=False,
            quiet=True,
        )
        cmd.execute(args)
        captured = capsys.readouterr()
        assert "Trace Replay Result" not in captured.out

    def test_replay_emits_terminal_summary(
        self, cmd, replayable_trace, tmp_path, capsys
    ):
        output_dir = str(tmp_path / "out")
        args = _replay_namespace(
            replayable_trace,
            output_dir,
            no_csv=True,
            json=False,
            quiet=False,
        )
        cmd.execute(args)
        out = capsys.readouterr().out
        assert "Trace Replay Result" in out


class TestCLIOnlyInstall:
    """Simulate the ``lmcache-cli``-only install: the heavy
    ``lmcache.v1.*`` imports in :mod:`lmcache.cli.commands.trace`
    failed at module load.  ``info`` / ``replay`` must surface a
    clear install hint instead of crashing.

    The install hint is printed directly to ``sys.stderr``; tests
    capture via ``capsys``.
    """

    def test_info_exits_with_install_hint(self, cmd, capsys, monkeypatch):
        # First Party
        import lmcache.cli.commands.trace as trace_mod

        monkeypatch.setattr(
            trace_mod,
            "_IMPORT_ERROR",
            ImportError("No module named 'lmcache.v1.distributed'"),
        )
        args = argparse.Namespace(trace_target="info", trace_path="/tmp/x.lct")
        with pytest.raises(SystemExit) as ei:
            cmd.execute(args)
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert "full LMCache package" in err
        assert "pip install lmcache" in err

    def test_replay_exits_with_install_hint(self, cmd, capsys, monkeypatch):
        # First Party
        import lmcache.cli.commands.trace as trace_mod

        monkeypatch.setattr(
            trace_mod,
            "_IMPORT_ERROR",
            ImportError("No module named 'lmcache.v1.mp_observability'"),
        )
        args = argparse.Namespace(trace_target="replay", trace_path="/tmp/x.lct")
        with pytest.raises(SystemExit) as ei:
            cmd.execute(args)
        assert ei.value.code == 2
        err = capsys.readouterr().err
        assert "full LMCache package" in err
