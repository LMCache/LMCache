# SPDX-License-Identifier: Apache-2.0
"""Tests for the anonymous usage telemetry package (lmcache/usage_telemetry/)."""

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
from lmcache import torch_device_type
from lmcache.usage_telemetry import (
    USAGE_SCHEMA_VERSION,
    ContinuousUsageContext,
    InitializeMPUsageContext,
    InitializeUsageContext,
    MPServerMessage,
    MPUsageContext,
    UsageContext,
    UsageMessageSender,
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.l1_usage import (
    InitializeL1Usage,
    L1UsageReporter,
)
from lmcache.usage_telemetry.l2_usage import (
    InitializeL2ConnectorUsage,
    L2ConnectorUsageReporter,
    L2TypeUsage,
)
from lmcache.usage_telemetry.metric_specs import MetricSpec
from lmcache.usage_telemetry.mp_continuous import (
    InitializeMPContinuousUsage,
    MPContinuousUsageReporter,
)
from lmcache.usage_telemetry.transport import usage_server_url
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdaptersConfig,
    get_type_name_for_config,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.multiprocess.config import MPServerConfig


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


def make_storage_manager_config() -> StorageManagerConfig:
    return StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 30,
                use_lazy=False,
                shm_name="test_shm",
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    )


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


class TestUsageServerUrl:
    @pytest.mark.parametrize(
        "base,expected",
        [
            ("http://stats.test", "http://stats.test/context"),
            ("http://stats.test/", "http://stats.test/context"),
            ("http://stats.test/api/v1", "http://stats.test/api/v1/context"),
        ],
    )
    def test_base_url_path_preserved(self, usage_env, monkeypatch, base, expected):
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_URL", base)
        assert usage_server_url("context") == expected


class TestOptOut:
    def test_enabled_by_default(self, usage_env):
        assert is_usage_tracking_enabled()

    def test_lmcache_track_usage_false(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
        assert not is_usage_tracking_enabled()

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE"])
    def test_do_not_track_env(self, usage_env, monkeypatch, value):
        monkeypatch.setenv("DO_NOT_TRACK", value)
        assert not is_usage_tracking_enabled()

    def test_do_not_track_env_unset_values(self, usage_env, monkeypatch):
        monkeypatch.setenv("DO_NOT_TRACK", "0")
        assert is_usage_tracking_enabled()


class TestIdentity:
    def test_identity_is_process_singleton(self, usage_env):
        first = get_usage_identity()
        second = get_usage_identity()
        assert first is second
        assert first.session_id
        assert first.machine_id

    def test_machine_id_persists_across_sessions(self, usage_env, monkeypatch):
        first = get_usage_identity()
        # Simulate a new process: the session singleton resets, the
        # machine_id file survives.
        monkeypatch.setattr("lmcache.usage_telemetry.identity._usage_identity", None)
        second = get_usage_identity()
        assert second.session_id != first.session_id
        assert second.machine_id == first.machine_id
        machine_id_file = usage_env / ".config" / "lmcache" / "machine_id"
        assert machine_id_file.read_text().strip() == first.machine_id

    def test_machine_id_empty_when_unwritable(self, usage_env):
        # Occupy the config-dir path with a regular file so the machine_id
        # file can be neither read nor created.
        (usage_env / ".config").mkdir()
        (usage_env / ".config" / "lmcache").touch()
        identity = get_usage_identity()
        assert identity.machine_id == ""
        assert identity.session_id


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


def publish_mp_traffic(bus: EventBus) -> None:
    """Publish one retrieve-end and one store-end event (4 + 2 chunks)."""
    bus.publish(
        Event(
            event_type=EventType.MP_RETRIEVE_END,
            metadata={
                "retrieved_count": 4,
                "device": f"{torch_device_type}:0",
                "engine_id": 1,
                "model_name": "test_model",
                "cache_salt": "",
                "total_bytes": 4096,
            },
        )
    )
    bus.publish(
        Event(
            event_type=EventType.MP_STORE_END,
            metadata={
                "stored_count": 2,
                "device": f"{torch_device_type}:0",
                "engine_id": 1,
                "model_name": "test_model",
                "total_bytes": 1000,
            },
        )
    )


class TestMPContinuous:
    def test_counters_flush_on_bus_stop(self, usage_env):
        sender = RecordingSender()
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        reporter = InitializeMPContinuousUsage(bus, chunk_size=256, sender=sender)
        assert reporter is not None

        publish_mp_traffic(bus)
        # stop() drains queued events, then the shutdown hook sends the
        # final flush.
        bus.stop()

        assert len(sender.sent) == 1
        url, payload = sender.sent[0]
        assert url == "http://stats.test/cache-usage"
        assert payload["message_type"] == "ContinuousContextMessage"
        assert payload["deployment_mode"] == "mp_server"
        assert payload["interval_num_hit_tokens"] == 4 * 256
        assert payload["interval_num_stored_tokens"] == 2 * 256
        assert payload["interval_stored_kv_size"] == 1000
        assert payload["sequence_number"] == 1
        assert payload["uptime_seconds"] >= 0
        assert payload["session_id"] == get_usage_identity().session_id

    def test_flush_resets_counters(self, usage_env):
        sender = RecordingSender()
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        reporter = InitializeMPContinuousUsage(bus, chunk_size=256, sender=sender)
        assert reporter is not None
        publish_mp_traffic(bus)
        bus.stop()

        reporter.flush()
        assert len(sender.sent) == 2
        assert sender.sent[1][1]["interval_num_hit_tokens"] == 0
        assert sender.sent[1][1]["interval_num_stored_tokens"] == 0
        assert sender.sent[1][1]["sequence_number"] == 2

    def test_initialize_returns_none_when_disabled(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
        bus = EventBus(EventBusConfig(enabled=True))
        assert InitializeMPContinuousUsage(bus, chunk_size=256) is None
        bus.stop()

    def test_flush_drops_counters_when_disabled(self, usage_env, monkeypatch):
        sender = RecordingSender()
        reporter = MPContinuousUsageReporter(chunk_size=256, sender=sender)
        monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
        reporter.flush()
        assert sender.sent == []
        reporter.shutdown()

    def test_raising_sender_does_not_break_flush(self, usage_env):
        reporter = MPContinuousUsageReporter(chunk_size=256, sender=RaisingSender())
        reporter.flush()
        reporter.shutdown()

    def test_malformed_interval_env_does_not_raise(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "10m")
        # Non-MP: constructed unguarded inside LMCacheStatsLogger.__init__,
        # so the constructor itself must not raise.
        context = ContinuousUsageContext.GetOrCreate(make_metadata())
        assert context.min_logging_interval == 600
        context.incr_or_send_stats(StubStats(interval_hit_tokens=1))
        # MP: falls back to the default interval instead of losing telemetry.
        sender = RecordingSender()
        reporter = MPContinuousUsageReporter(chunk_size=256, sender=sender)
        reporter.flush()
        assert sender.sent
        reporter.shutdown()

    def test_specs_must_cover_message_fields(self, usage_env):
        incomplete = [
            MetricSpec(
                event_type=EventType.MP_STORE_END,
                field="interval_stored_kv_size",
                extract=lambda e: int(e.metadata["total_bytes"]),
                reduce=sum,
            )
        ]
        with pytest.raises(ValueError, match="exactly once"):
            MPContinuousUsageReporter(chunk_size=256, specs=incomplete)

    def test_custom_reduce_function(self, usage_env):
        def max_or_zero(samples):
            return max(samples, default=0)

        specs = [
            MetricSpec(
                event_type=EventType.MP_RETRIEVE_END,
                field="interval_num_hit_tokens",
                extract=lambda e: int(e.metadata["retrieved_count"]),
                reduce=sum,
            ),
            MetricSpec(
                event_type=EventType.MP_STORE_END,
                field="interval_num_stored_tokens",
                extract=lambda e: int(e.metadata["stored_count"]),
                reduce=sum,
            ),
            # Largest single store of the interval instead of the total.
            MetricSpec(
                event_type=EventType.MP_STORE_END,
                field="interval_stored_kv_size",
                extract=lambda e: int(e.metadata["total_bytes"]),
                reduce=max_or_zero,
            ),
        ]
        sender = RecordingSender()
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        reporter = MPContinuousUsageReporter(chunk_size=256, sender=sender, specs=specs)
        bus.register_subscriber(reporter)
        publish_mp_traffic(bus)
        publish_mp_traffic(bus)
        bus.stop()

        assert len(sender.sent) == 1
        payload = sender.sent[0][1]
        assert payload["interval_num_hit_tokens"] == 8
        assert payload["interval_num_stored_tokens"] == 4
        assert payload["interval_stored_kv_size"] == 1000  # max, not 2000

    def test_buffer_overflow_triggers_early_flush(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_USAGE_TRACK_INTERVAL", "3")
        sender = RecordingSender()
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        reporter = MPContinuousUsageReporter(
            chunk_size=256, sender=sender, max_buffered_samples=2
        )
        bus.register_subscriber(reporter)
        # Wake requests during the flush thread's initial wait are
        # deferred to the first run, so let the first periodic flush
        # pass before testing the overflow path.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not sender.sent:
            time.sleep(0.01)
        assert sender.sent, "no first periodic flush"
        baseline = len(sender.sent)
        # The second round of traffic fills a 2-sample buffer and wakes
        # the flush thread: a new message arrives well before the next
        # 3 s tick.
        publish_mp_traffic(bus)
        publish_mp_traffic(bus)
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline and len(sender.sent) == baseline:
            time.sleep(0.01)
        assert len(sender.sent) > baseline, "overflow did not wake the flush thread"
        # The early flush can race the drain thread and split the samples
        # across messages; stop() flushes the rest, so assert the totals
        # across all sent messages.
        bus.stop()
        assert (
            sum(
                int(payload["interval_num_stored_tokens"]) for _, payload in sender.sent
            )
            == 2 * 2 * 256
        )
        assert (
            sum(int(payload["interval_num_hit_tokens"]) for _, payload in sender.sent)
            == 2 * 4 * 256
        )


def publish_l2_traffic(bus: EventBus) -> None:
    """Publish L2 store/load events: dax traffic twice, posix once."""
    for stored_bytes, ok, failed in ((1000, 3, 1), (500, 2, 0)):
        bus.publish(
            Event(
                event_type=EventType.L2_STORE_COMPLETED,
                metadata={
                    "adapter_index": 0,
                    "task_id": 1,
                    "l2_name": "dax",
                    "bytes_transferred": stored_bytes,
                    "succeeded_count": ok,
                    "failed_count": failed,
                },
            )
        )
    bus.publish(
        Event(
            event_type=EventType.L2_LOAD_TASK_SUBMITTED,
            metadata={
                "request_id": 7,
                "adapter_index": 0,
                "task_id": 2,
                "l2_name": "dax",
                "key_count": 4,
                "total_bytes": 2048,
            },
        )
    )
    bus.publish(
        Event(
            event_type=EventType.L2_STORE_COMPLETED,
            metadata={
                "adapter_index": 1,
                "task_id": 3,
                "l2_name": "posix",
                "bytes_transferred": 256,
                "succeeded_count": 1,
                "failed_count": 0,
            },
        )
    )


class StubStorageManager:
    """Duck-typed StorageManager exposing only the usage accessor."""

    def __init__(self, usages_by_type: dict[str, list[AdapterUsage]]) -> None:
        self._usages_by_type = usages_by_type

    def get_l2_usages_by_type(self) -> dict[str, list[AdapterUsage]]:
        return self._usages_by_type


class TestL2ConnectorUsage:
    def _messages_by_type(
        self, sender: RecordingSender
    ) -> dict[str, dict[str, object]]:
        return {str(payload["l2_name"]): payload for _, payload in sender.sent}

    def test_traffic_and_occupancy_flush_on_bus_stop(self, usage_env):
        sender = RecordingSender()
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        reporter = L2ConnectorUsageReporter(
            usage_probe=lambda: {"dax": L2TypeUsage(100, 1000, 0)}, sender=sender
        )
        bus.register_subscriber(reporter)
        publish_l2_traffic(bus)
        bus.stop()  # drains, then the shutdown hook sends the final flush

        by_type = self._messages_by_type(sender)
        assert set(by_type) == {"dax", "posix"}
        for _, payload in sender.sent:
            assert payload["message_type"] == "L2ConnectorUsageMessage"
            assert payload["deployment_mode"] == "mp_server"
            assert payload["sequence_number"] == 1
            assert payload["active_seconds"] > 0

        dax = by_type["dax"]
        assert dax["interval_stored_bytes"] == 1500
        assert dax["interval_store_succeeded_keys"] == 5
        assert dax["interval_store_failed_keys"] == 1
        assert dax["interval_load_submitted_keys"] == 4
        assert dax["interval_load_submitted_bytes"] == 2048
        assert dax["bytes_used"] == 100
        assert dax["capacity_bytes"] == 1000

        # posix had traffic but was absent from the probe (removed
        # mid-interval): occupancy is the unavailable sentinel.
        posix = by_type["posix"]
        assert posix["interval_stored_bytes"] == 256
        assert posix["bytes_used"] == -1

        assert sender.sent[0][0].endswith("l2-usage")

    def test_idle_present_type_reports_presence(self, usage_env):
        sender = RecordingSender()
        reporter = L2ConnectorUsageReporter(
            usage_probe=lambda: {"dax": L2TypeUsage(50, 200, 0)}, sender=sender
        )
        reporter.flush()
        assert len(sender.sent) == 1
        payload = sender.sent[0][1]
        assert payload["l2_name"] == "dax"
        assert payload["interval_stored_bytes"] == 0
        assert payload["bytes_used"] == 50
        reporter.shutdown()

    def test_no_adapters_sends_nothing(self, usage_env):
        sender = RecordingSender()
        reporter = L2ConnectorUsageReporter(usage_probe=dict, sender=sender)
        reporter.flush()
        assert sender.sent == []
        reporter.shutdown()

    def test_probe_failure_keeps_traffic_with_sentinel(self, usage_env):
        def broken_probe() -> dict[str, L2TypeUsage]:
            raise RuntimeError("probe failure")

        sender = RecordingSender()
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        reporter = L2ConnectorUsageReporter(usage_probe=broken_probe, sender=sender)
        bus.register_subscriber(reporter)
        publish_l2_traffic(bus)
        bus.stop()

        by_type = self._messages_by_type(sender)
        assert set(by_type) == {"dax", "posix"}
        assert by_type["dax"]["interval_stored_bytes"] == 1500
        assert by_type["dax"]["bytes_used"] == -1

    def test_initialize_aggregates_bounded_and_unbounded(self, usage_env):
        storage_manager = StubStorageManager(
            {
                "dax": [
                    AdapterUsage(total_bytes_used=100, total_capacity_bytes=1000),
                    AdapterUsage(total_bytes_used=25, total_capacity_bytes=0),
                ]
            }
        )
        sender = RecordingSender()
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        reporter = InitializeL2ConnectorUsage(bus, storage_manager, sender=sender)
        assert reporter is not None
        bus.stop()

        payload = sender.sent[0][1]
        assert payload["l2_name"] == "dax"
        assert payload["bytes_used"] == 125
        assert payload["capacity_bytes"] == 1000
        assert payload["unbounded_adapters"] == 1

    def test_initialize_returns_none_when_disabled(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
        bus = EventBus(EventBusConfig(enabled=True))
        assert InitializeL2ConnectorUsage(bus, StubStorageManager({})) is None
        bus.stop()


class StubL1StorageManager:
    """Duck-typed StorageManager exposing only the L1 usage accessor."""

    def __init__(self, used: int, total: int) -> None:
        self._used = used
        self._total = total

    def get_l1_usage(self) -> tuple[int, int]:
        return self._used, self._total


class TestL1Usage:
    def test_flush_sends_occupancy(self, usage_env):
        sender = RecordingSender()
        reporter = L1UsageReporter(usage_probe=lambda: (512, 4096), sender=sender)
        reporter.flush()
        assert len(sender.sent) == 1
        url, payload = sender.sent[0]
        assert url.endswith("l1-usage")
        assert payload["message_type"] == "L1UsageMessage"
        assert payload["deployment_mode"] == "mp_server"
        assert payload["bytes_used"] == 512
        assert payload["capacity_bytes"] == 4096
        assert payload["sequence_number"] == 1
        assert payload["active_seconds"] > 0
        reporter.shutdown()

    def test_probe_failure_sends_sentinels(self, usage_env):
        def broken_probe() -> tuple[int, int]:
            raise RuntimeError("probe failure")

        sender = RecordingSender()
        reporter = L1UsageReporter(usage_probe=broken_probe, sender=sender)
        reporter.flush()
        payload = sender.sent[0][1]
        assert payload["bytes_used"] == -1
        assert payload["capacity_bytes"] == 0
        reporter.shutdown()

    def test_bus_stop_sends_final_flush(self, usage_env):
        sender = RecordingSender()
        bus = EventBus(EventBusConfig(enabled=True))
        bus.start()
        reporter = InitializeL1Usage(
            bus, StubL1StorageManager(100, 1000), sender=sender
        )
        assert reporter is not None
        bus.stop()
        assert len(sender.sent) == 1
        assert sender.sent[0][1]["bytes_used"] == 100

    def test_initialize_returns_none_when_disabled(self, usage_env, monkeypatch):
        monkeypatch.setenv("LMCACHE_TRACK_USAGE", "false")
        bus = EventBus(EventBusConfig(enabled=True))
        assert InitializeL1Usage(bus, StubL1StorageManager(0, 0)) is None
        bus.stop()


class TestMPUsage:
    def test_mp_server_message_from_configs(self, usage_env):
        message = MPServerMessage.from_configs(
            MPServerConfig(), make_storage_manager_config()
        )
        assert message.chunk_size == 256
        assert message.hash_algorithm == "blake3"
        assert message.engine_type == "default"
        assert message.supported_transfer_mode == "lmcache_driven"
        assert not message.p2p_enabled
        assert message.l1_size_bytes == 1 << 30
        assert message.l1_medium == "dram"
        assert message.l1_shm_enabled
        assert message.eviction_policy == "LRU"
        assert message.l2_adapter_types == ""
        assert message.l2_serde_types == ""
        assert message.l2_store_policy == "default"
        assert message.l2_prefetch_policy == "default"
        assert not message.enable_segmented_prefix
        assert message.lmcache_version

    def test_mp_server_message_serde_and_segmented_prefix(self, usage_env, tmp_path):
        # The FS adapter needs the native storage ops extension; skip on
        # builds without it.
        fs_l2_adapter = pytest.importorskip(
            "lmcache.v1.distributed.l2_adapters.fs_l2_adapter",
            reason="requires lmcache.lmcache_native",
        )
        serde = pytest.importorskip(
            "lmcache.v1.distributed.serde",
            reason="requires lmcache.lmcache_native",
        )
        fs_config = fs_l2_adapter.FSL2AdapterConfig(
            base_path=str(tmp_path),
            relative_tmp_dir=None,
            read_ahead_size=None,
            use_odirect=False,
        )
        fs_config.serde_config = serde.SerdeConfig(type="fp8", kwargs={})
        storage_config = make_storage_manager_config()
        storage_config.l2_adapter_config = L2AdaptersConfig([fs_config])

        message = MPServerMessage.from_configs(
            MPServerConfig(enable_segmented_prefix=True), storage_config
        )
        assert message.enable_segmented_prefix
        assert message.l2_adapter_types == get_type_name_for_config(fs_config)
        assert message.l2_serde_types == "fp8"

    def test_mp_server_message_gds_l1(self, usage_env, tmp_path):
        storage_config = StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=1 << 30,
                    use_lazy=False,
                    shm_name="",
                ),
                gds_l1_config=GdsL1Config(
                    file_location=str(tmp_path), size_in_bytes=2 << 30
                ),
            ),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )
        message = MPServerMessage.from_configs(MPServerConfig(), storage_config)
        assert message.l1_medium == "gds"
        assert message.l1_size_bytes == 2 << 30
        assert not message.l1_shm_enabled

    def test_mp_usage_context_sends_messages(self, usage_env):
        sender = RecordingSender()
        context = MPUsageContext(
            MPServerConfig(),
            make_storage_manager_config(),
            sender=sender,
        )
        context.report_once()

        message_types = [payload["message_type"] for _, payload in sender.sent]
        assert message_types == ["EnvMessage", "MPServerMessage"]

        identity = get_usage_identity()
        for url, payload in sender.sent:
            assert url == "http://stats.test/context"
            assert payload["schema_version"] == USAGE_SCHEMA_VERSION
            assert payload["session_id"] == identity.session_id
            assert payload["deployment_mode"] == "mp_server"

        mp_payload = sender.sent[1][1]
        assert mp_payload["chunk_size"] == 256
        assert "instance_id" not in mp_payload

    def test_initialize_mp_returns_none_when_disabled(self, usage_env, monkeypatch):
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        context = InitializeMPUsageContext(
            MPServerConfig(), make_storage_manager_config()
        )
        assert context is None


class RaisingSender(UsageMessageSender):
    """Transport stub whose every send raises."""

    def send(self, url: str, payload: dict[str, object]) -> None:
        raise RuntimeError("telemetry transport failure")


class TestFailureIsolation:
    def test_swallow_telemetry_errors_returns_none(self):
        @swallow_telemetry_errors
        def boom() -> int:
            raise ValueError("telemetry bug")

        assert boom() is None

    def test_raising_sender_does_not_break_mp_report(self, usage_env):
        context = MPUsageContext(
            MPServerConfig(), make_storage_manager_config(), sender=RaisingSender()
        )
        context.report_once()

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

    def test_default_sender_swallows_unreachable_server(self, usage_env):
        # Port 1 is never listening; the send must not raise.
        UsageMessageSender().send("http://127.0.0.1:1/context", {"k": "v"})

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

    def test_mp_report(self, usage_sink):
        context = InitializeMPUsageContext(
            MPServerConfig(), make_storage_manager_config()
        )
        assert context is not None
        received = usage_sink.wait_for(2)
        assert [payload["message_type"] for _, payload in received] == [
            "EnvMessage",
            "MPServerMessage",
        ]
        for path, payload in received:
            assert path == "/context"
            assert payload["deployment_mode"] == "mp_server"
        assert len({payload["session_id"] for _, payload in received}) == 1

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
