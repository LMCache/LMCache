# SPDX-License-Identifier: Apache-2.0
"""Tests for the single-process (InProcess) usage telemetry stack.

The MP-side tests live in ``tests/test_usage_telemetry.py``; this file keeps the
InProcess-only reporter tests out of that MP module, so the MP module never
imports ``lmcache.inprocess``.
"""

# Standard
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.inprocess.usage_telemetry.context import (
    InitializeUsageContext,
    UsageContext,
)
from lmcache.inprocess.usage_telemetry.continuous import ContinuousUsageContext
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.usage_telemetry import (
    USAGE_SCHEMA_VERSION,
    UsageMessageSender,
    get_usage_identity,
)


class RecordingSender(UsageMessageSender):
    """Transport stub that records payloads instead of POSTing them."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, object]]] = []

    def send(self, url: str, payload: dict[str, object]) -> None:
        self.sent.append((url, payload))


@dataclass
class StubStats:
    """Minimal stand-in for LMCacheStats used by incr_or_send_stats."""

    interval_hit_tokens: int = 0
    interval_stored_tokens: int = 0
    interval_request_cache_lifespan: list[float] = field(default_factory=list)


class UsageSink(ThreadingHTTPServer):
    """In-process HTTP server recording every POSTed usage payload."""

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _UsageSinkHandler)
        self.received: list[tuple[str, dict[str, object]]] = []

    def wait_for(
        self, count: int, timeout: float = 15.0
    ) -> list[tuple[str, dict[str, object]]]:
        """Return received (path, payload) pairs once *count* arrived."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and len(self.received) < count:
            time.sleep(0.01)
        return list(self.received)


class _UsageSinkHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.received.append((self.path, body))  # type: ignore[attr-defined]
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def usage_env(monkeypatch, tmp_path):
    """Isolate usage-telemetry state: HOME, env vars, and singletons."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LMCACHE_USAGE_TRACK_URL", "http://stats.test")
    monkeypatch.delenv("LMCACHE_TRACK_USAGE", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("LMCACHE_USAGE_TRACK_INTERVAL", raising=False)
    monkeypatch.setattr("lmcache.usage_telemetry.identity._usage_identity", None)
    monkeypatch.setattr(ContinuousUsageContext, "_instance", None)
    return tmp_path


@pytest.fixture
def usage_sink(usage_env, monkeypatch):
    """Local HTTP sink; points LMCACHE_USAGE_TRACK_URL at it."""
    sink = UsageSink()
    thread = threading.Thread(target=sink.serve_forever, daemon=True)
    thread.start()
    port = sink.server_address[1]
    monkeypatch.setenv("LMCACHE_USAGE_TRACK_URL", f"http://127.0.0.1:{port}")
    yield sink
    sink.shutdown()
    sink.server_close()


def make_metadata() -> LMCacheMetadata:
    return LMCacheMetadata(
        model_name="test_model",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.bfloat16,
        kv_shape=(32, 2, 256, 32, 128),
        use_mla=False,
        role="worker",
    )


class TestUsageContext:
    def test_report_once_sends_all_messages(self, usage_env):
        sender = RecordingSender()
        context = UsageContext(
            LMCacheEngineConfig.from_defaults(),
            make_metadata(),
            sender=sender,
        )
        context.report_once()

        message_types = [payload["message_type"] for _, payload in sender.sent]
        assert message_types == ["EnvMessage", "EngineMessage", "MetadataMessage"]

        identity = get_usage_identity()
        for url, payload in sender.sent:
            assert url == "http://stats.test/context"
            assert payload["schema_version"] == USAGE_SCHEMA_VERSION
            assert payload["session_id"] == identity.session_id
            assert payload["machine_id"] == identity.machine_id
            assert payload["deployment_mode"] == "single_process"

        engine_payload = sender.sent[1][1]
        assert engine_payload["model_name"] == "test_model"
        assert engine_payload["kv_dtype"] == "torch.bfloat16"

    def test_local_log_written(self, usage_env, tmp_path):
        log_path = tmp_path / "usage.log"
        context = UsageContext(
            LMCacheEngineConfig.from_defaults(),
            make_metadata(),
            local_log=str(log_path),
            sender=RecordingSender(),
        )
        context.report_once()
        content = log_path.read_text()
        assert "message_type: EnvMessage" in content
        assert "message_type: EngineMessage" in content
        assert "session_id:" in content


class TestInitializeUsageContext:
    def test_returns_none_when_disabled(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
        context = InitializeUsageContext(
            LMCacheEngineConfig.from_defaults(), make_metadata()
        )
        assert context is None

    def test_reports_from_background_thread(self, usage_env):
        sender = RecordingSender()
        context = InitializeUsageContext(
            LMCacheEngineConfig.from_defaults(), make_metadata(), sender=sender
        )
        assert context is not None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and len(sender.sent) < 3:
            time.sleep(0.01)
        assert len(sender.sent) == 3


class TestContinuousUsageContext:
    def test_flush_and_reset(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "0")
        sender = RecordingSender()
        context = ContinuousUsageContext(make_metadata(), sender=sender)

        context.incr_or_send_stats(
            StubStats(
                interval_hit_tokens=100,
                interval_stored_tokens=200,
                interval_request_cache_lifespan=[0.5, 2.0],
            )
        )

        assert len(sender.sent) == 2
        usage_url, usage_payload = sender.sent[0]
        assert usage_url.endswith("cache-usage")
        assert usage_payload["message_type"] == "ContinuousContextMessage"
        assert usage_payload["interval_num_hit_tokens"] == 100
        assert usage_payload["interval_num_stored_tokens"] == 200
        assert usage_payload["sequence_number"] == 1
        assert usage_payload["session_id"] == get_usage_identity().session_id
        assert usage_payload["deployment_mode"] == "single_process"
        assert usage_payload["uptime_seconds"] >= 0

        lifespan_url, lifespan_payload = sender.sent[1]
        assert lifespan_url.endswith("cache-lifespan")
        assert lifespan_payload["message_type"] == "CacheLifespanMessage"
        assert lifespan_payload["sequence_number"] == 1
        assert lifespan_payload["uptime_seconds"] >= 0

        # Counters reset after the flush; a second flush reports zeros with
        # the next sequence number.
        context.incr_or_send_stats(StubStats())
        assert sender.sent[2][1]["interval_num_hit_tokens"] == 0
        assert sender.sent[2][1]["sequence_number"] == 2

    def test_disabled_is_noop(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "0")
        sender = RecordingSender()
        context = ContinuousUsageContext(make_metadata(), sender=sender)
        context.incr_or_send_stats(
            StubStats(
                interval_hit_tokens=100,
                interval_request_cache_lifespan=[1.0],
            )
        )
        assert sender.sent == []
        assert context.cache_lifespan_data == []

    def test_list_to_histogram(self, usage_env):
        context = ContinuousUsageContext(make_metadata(), sender=RecordingSender())
        histogram = context.list_to_histogram([0.5, 2.0, 3.0], [0, 1, 5, 10])
        assert histogram == {0: 0, 1: 1, 5: 2, 10: 0}

    def test_malformed_interval_env_does_not_raise(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "10m")
        # Constructed unguarded inside LMCacheStatsLogger.__init__, so the
        # constructor itself must not raise.
        context = ContinuousUsageContext.GetOrCreate(make_metadata())
        assert context.min_logging_interval == 600
        context.incr_or_send_stats(StubStats(interval_hit_tokens=1))


class RaisingSender(UsageMessageSender):
    """Transport stub whose every send raises."""

    def send(self, url: str, payload: dict[str, object]) -> None:
        raise RuntimeError("telemetry transport failure")


class TestFailureIsolation:
    def test_raising_sender_does_not_break_continuous_flush(
        self, usage_env, monkeypatch
    ):
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "0")
        context = ContinuousUsageContext(make_metadata(), sender=RaisingSender())
        context.incr_or_send_stats(StubStats(interval_hit_tokens=1))

    def test_unwritable_local_log_does_not_break_report(self, usage_env, tmp_path):
        context = UsageContext(
            LMCacheEngineConfig.from_defaults(),
            make_metadata(),
            local_log=str(tmp_path),  # a directory: open() for append fails
            sender=RecordingSender(),
        )
        context.report_once()

    def test_nonstandard_kv_shape_degrades_to_zero_bytes(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "0")
        metadata = make_metadata()
        metadata.kv_shape = ()
        sender = RecordingSender()
        context = ContinuousUsageContext(metadata, sender=sender)
        context.incr_or_send_stats(StubStats(interval_stored_tokens=100))
        assert sender.sent[0][1]["interval_stored_kv_size"] == 0


class TestSingletons:
    def test_get_or_create_returns_same_instance(self, usage_env):
        metadata = make_metadata()
        first = ContinuousUsageContext.GetOrCreate(metadata)
        second = ContinuousUsageContext.GetOrCreate(metadata)
        assert first is second

    def test_get_or_create_keeps_first_instance_on_metadata_mismatch(self, usage_env):
        first = ContinuousUsageContext.GetOrCreate(make_metadata())
        other_metadata = make_metadata()
        other_metadata.model_name = "another_model"
        assert ContinuousUsageContext.GetOrCreate(other_metadata) is first


class TestEndToEnd:
    """Exercise the real HTTP transport against a local sink (no stubs)."""

    def test_single_process_report(self, usage_sink):
        context = InitializeUsageContext(
            LMCacheEngineConfig.from_defaults(), make_metadata()
        )
        assert context is not None
        received = usage_sink.wait_for(3)
        assert len(received) == 3
        assert {path for path, _ in received} == {"/context"}
        assert [payload["message_type"] for _, payload in received] == [
            "EnvMessage",
            "EngineMessage",
            "MetadataMessage",
        ]
        for _, payload in received:
            assert payload["deployment_mode"] == "single_process"

    def test_continuous_report(self, usage_sink, monkeypatch):
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "0")
        context = ContinuousUsageContext(make_metadata())
        context.incr_or_send_stats(
            StubStats(
                interval_hit_tokens=7,
                interval_stored_tokens=9,
                interval_request_cache_lifespan=[1.0],
            )
        )
        received = usage_sink.wait_for(2)
        assert [path for path, _ in received] == ["/cache-usage", "/cache-lifespan"]
        assert received[0][1]["interval_num_hit_tokens"] == 7
        # Histogram keys are bucket bounds (stringified by JSON); the 1.0 s
        # sample falls in the [1, 5) bin, keyed "5".
        assert received[1][1]["cache_lifespan_histogram"]["5"] == 1
