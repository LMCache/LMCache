# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``lmcache trace`` CLI command.

End-to-end: ``info`` is smoke-tested against a tiny trace file written
through the real recorder.  ``replay`` is exercised against the same
fixture via the driver (argparse wiring only — the driver itself has
its own tests in ``tests/tools/trace_replay``).  ``record`` is a stub
and simply asserts its non-zero exit code.
"""

# Future
from __future__ import annotations

# Standard
import argparse
import time

# Third Party
import pytest

# First Party
from lmcache.cli.commands.trace import TraceCommand
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace.decorator import (
    publish_call_event,
    set_tracing_enabled,
)
from lmcache.v1.mp_observability.trace.recorder import StorageTraceRecorder
import lmcache.v1.mp_observability.event_bus as _bus_module


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
                "--pace",
                "asap",
            ]
        )
        assert args.trace_target == "replay"
        assert args.pace == "asap"
        assert args.l1_size_gb == 0.0625
        assert args.eviction_policy == "LRU"

    def test_record_parses(self, parser):
        args = parser.parse_args(["trace", "record"])
        assert args.trace_target == "record"


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


class TestRecordSubcommandStub:
    def test_record_exits_with_code_2(self, cmd, capsys):
        args = argparse.Namespace(trace_target="record")
        with pytest.raises(SystemExit) as ei:
            cmd.execute(args)
        assert ei.value.code == 2
        out = capsys.readouterr().out
        assert "lmcache server" in out
        assert "--trace-level storage" in out
