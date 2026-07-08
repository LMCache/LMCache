# SPDX-License-Identifier: Apache-2.0
"""Tests for the anonymous usage telemetry package (lmcache/usage_telemetry/)."""

# Standard
from dataclasses import dataclass, field
import time

# Third Party
import pytest
import torch

# First Party
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
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.metadata import LMCacheMetadata
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


@pytest.fixture
def usage_env(monkeypatch, tmp_path):
    """Isolate usage-telemetry state: HOME, env vars, and singletons."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("LMCACHE_TRACK_USAGE", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.delenv("LMCACHE_USAGE_TRACK_INTERVAL", raising=False)
    monkeypatch.setattr("lmcache.usage_telemetry.identity._usage_identity", None)
    monkeypatch.setattr(ContinuousUsageContext, "_instance", None)
    return tmp_path


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

    def test_do_not_track_file(self, usage_env):
        marker = usage_env / ".config" / "lmcache" / "do_not_track"
        marker.parent.mkdir(parents=True)
        marker.touch()
        assert not is_usage_tracking_enabled()


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
            "http://stats.test/context",
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

        engine_payload = sender.sent[1][1]
        assert engine_payload["model_name"] == "test_model"
        assert engine_payload["kv_dtype"] == "torch.bfloat16"

    def test_local_log_written(self, usage_env, tmp_path):
        log_path = tmp_path / "usage.log"
        context = UsageContext(
            "http://stats.test/context",
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

        lifespan_url, lifespan_payload = sender.sent[1]
        assert lifespan_url.endswith("cache-lifespan")
        assert lifespan_payload["message_type"] == "CacheLifespanMessage"
        assert lifespan_payload["sequence_number"] == 1

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


class TestMPUsage:
    def test_mp_server_message_from_configs(self, usage_env):
        message = MPServerMessage.from_configs(
            MPServerConfig(), make_storage_manager_config()
        )
        assert message.chunk_size == 256
        assert message.hash_algorithm == "blake3"
        assert message.engine_type == "default"
        assert message.supported_transfer_mode == "auto"
        assert not message.p2p_enabled
        assert message.l1_size_bytes == 1 << 30
        assert message.l1_medium == "dram"
        assert message.l1_shm_enabled
        assert message.eviction_policy == "LRU"
        assert message.l2_adapter_types == ""
        assert message.l2_store_policy == "default"
        assert message.l2_prefetch_policy == "default"
        assert message.lmcache_version

    def test_mp_usage_context_sends_messages(self, usage_env):
        sender = RecordingSender()
        context = MPUsageContext(
            "http://stats.test/context",
            MPServerConfig(),
            make_storage_manager_config(),
            sender=sender,
        )
        context.report_once()

        message_types = [payload["message_type"] for _, payload in sender.sent]
        assert message_types == ["EnvMessage", "MPServerMessage"]

        identity = get_usage_identity()
        for _, payload in sender.sent:
            assert payload["schema_version"] == USAGE_SCHEMA_VERSION
            assert payload["session_id"] == identity.session_id

        mp_payload = sender.sent[1][1]
        assert mp_payload["chunk_size"] == 256
        assert "instance_id" not in mp_payload

    def test_initialize_mp_returns_none_when_disabled(self, usage_env, monkeypatch):
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        context = InitializeMPUsageContext(
            MPServerConfig(), make_storage_manager_config()
        )
        assert context is None
