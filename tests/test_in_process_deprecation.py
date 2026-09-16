# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the in-process runtime deprecation warning."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
import logging
import os
import subprocess
import sys

# Third Party
import pytest

# First Party
from lmcache import deprecation

pytestmark = pytest.mark.no_shared_allocator


class _RecordHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def deprecation_records(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[logging.LogRecord]]:
    """Capture the non-propagating deprecation logger with fresh guard state."""
    monkeypatch.setattr(deprecation, "_warned_pid", None)
    handler = _RecordHandler()
    original_level = deprecation.logger.level
    deprecation.logger.setLevel(logging.WARNING)
    deprecation.logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        deprecation.logger.removeHandler(handler)
        deprecation.logger.setLevel(original_level)


def _deprecation_messages(records: list[logging.LogRecord]) -> list[str]:
    return [
        record.getMessage()
        for record in records
        if record.getMessage() == deprecation.IN_PROCESS_DEPRECATION_MESSAGE
    ]


def _construct_legacy_engine(monkeypatch: pytest.MonkeyPatch) -> object:
    # First Party
    from lmcache.v1 import cache_engine

    monkeypatch.setattr(
        cache_engine.multiprocessing, "set_start_method", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(cache_engine, "InitializeUsageContext", lambda *_args: None)
    monkeypatch.setattr(
        cache_engine,
        "LMCStatsMonitor",
        SimpleNamespace(GetOrCreate=lambda: object()),
    )
    monkeypatch.setattr(
        cache_engine,
        "PinMonitor",
        SimpleNamespace(GetOrCreate=lambda *_args: object()),
    )
    monkeypatch.setattr(cache_engine.gc, "collect", lambda: 0)

    config = SimpleNamespace(
        get_extra_config_value=lambda *_args: False,
        enable_controller=False,
        get_lmcache_worker_ids=lambda *_args: [],
        enable_async_loading=False,
        use_layerwise=False,
        enable_kv_events=False,
        enable_pd=False,
        pd_role=None,
        store_location=None,
        retrieve_locations=None,
        enable_blending=False,
        py_enable_gc=True,
        enable_hidden_state_cache=False,
    )
    metadata = SimpleNamespace(
        use_mla=False,
        world_size=1,
        role="worker",
        worker_id=0,
        kv_shape=(1, 2, 1, 1, 1),
    )
    return cache_engine.LMCacheEngine(
        config,  # type: ignore[arg-type]
        metadata,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        None,
        lambda *_args: None,
        lambda *_args: None,
    )


def _construct_vllm_scheduler(monkeypatch: pytest.MonkeyPatch) -> object:
    pytest.importorskip("vllm")

    # First Party
    from lmcache.integration.vllm import vllm_v1_adapter

    config = vllm_v1_adapter.LMCacheEngineConfig.from_defaults()
    manager = SimpleNamespace(
        lmcache_engine=None,
        start_services=lambda: None,
    )
    monkeypatch.setattr(vllm_v1_adapter, "print_banner_once", lambda *_args: None)
    monkeypatch.setattr(vllm_v1_adapter, "lmcache_get_or_create_config", lambda: config)
    monkeypatch.setattr(vllm_v1_adapter, "VllmServiceFactory", lambda *_args: object())
    monkeypatch.setattr(
        vllm_v1_adapter, "LMCacheManager", lambda *_args, **_kwargs: manager
    )
    monkeypatch.setattr(
        vllm_v1_adapter.LMCacheConnectorV1Impl,
        "_init_connector_state",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        vllm_v1_adapter.LMCacheConnectorV1Impl,
        "_setup_metrics",
        lambda *_args: None,
    )
    monkeypatch.setattr(vllm_v1_adapter.utils, "get_version", lambda: "test")

    vllm_config = SimpleNamespace(
        device_config=SimpleNamespace(device="cpu"),
        kv_transfer_config=SimpleNamespace(
            kv_role="kv_both",
            kv_connector_extra_config=None,
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    connector = vllm_v1_adapter.LMCacheConnectorV1Impl(
        vllm_config,  # type: ignore[arg-type]
        vllm_v1_adapter.KVConnectorRole.SCHEDULER,
        object(),  # type: ignore[arg-type]
    )
    assert connector.lmcache_engine is None
    return connector


def test_legacy_engine_warns_once_for_repeated_initialization(
    monkeypatch: pytest.MonkeyPatch,
    deprecation_records: list[logging.LogRecord],
) -> None:
    _construct_legacy_engine(monkeypatch)
    _construct_legacy_engine(monkeypatch)

    assert _deprecation_messages(deprecation_records) == [
        deprecation.IN_PROCESS_DEPRECATION_MESSAGE
    ]


def test_vllm_scheduler_warns_without_engine_and_shared_guard_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
    deprecation_records: list[logging.LogRecord],
) -> None:
    _construct_vllm_scheduler(monkeypatch)
    _construct_legacy_engine(monkeypatch)

    assert _deprecation_messages(deprecation_records) == [
        deprecation.IN_PROCESS_DEPRECATION_MESSAGE
    ]


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_forked_child_emits_its_own_warning(
    deprecation_records: list[logging.LogRecord],
) -> None:
    read_fd, write_fd = os.pipe()

    class _PipeHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.getMessage() == deprecation.IN_PROCESS_DEPRECATION_MESSAGE:
                os.write(write_fd, b"warning\n")

    handler = _PipeHandler(logging.WARNING)
    deprecation.logger.addHandler(handler)
    try:
        deprecation.warn_in_process_mode_deprecated()
        deprecation.warn_in_process_mode_deprecated()
        child_pid = os.fork()
        if child_pid == 0:
            os.close(read_fd)
            try:
                deprecation.warn_in_process_mode_deprecated()
            except BaseException:
                os._exit(1)
            os.close(write_fd)
            os._exit(0)

        os.close(write_fd)
        _, status = os.waitpid(child_pid, 0)
        output = os.read(read_fd, 1024)
    finally:
        deprecation.logger.removeHandler(handler)
        try:
            os.close(read_fd)
        except OSError:
            pass
        try:
            os.close(write_fd)
        except OSError:
            pass

    assert os.waitstatus_to_exitcode(status) == 0
    assert output.splitlines() == [b"warning", b"warning"]
    assert _deprecation_messages(deprecation_records) == [
        deprecation.IN_PROCESS_DEPRECATION_MESSAGE
    ]


def test_imports_shared_objects_and_mp_adapters_are_silent() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("vllm")
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import torch
from lmcache.integration.sglang.sglang_adapter import LoadMetadata, StoreMetadata
import lmcache.integration.sglang.multi_process_adapter
import lmcache.integration.vllm.vllm_multi_process_adapter
import lmcache.integration.vllm.vllm_v1_adapter
import lmcache.v1.cache_engine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.metadata import LMCacheMetadata

LMCacheEngineConfig.from_defaults()
LMCacheMetadata(
    model_name="test",
    world_size=1,
    local_world_size=1,
    worker_id=0,
    local_worker_id=0,
    kv_dtype=torch.float16,
    kv_shape=(1, 2, 1, 1, 1),
)
LoadMetadata(
    token_ids=[],
    slot_mapping=torch.empty(0, dtype=torch.int64),
    offset=0,
    request_id="request",
)
StoreMetadata(
    last_node=None,
    token_ids=[],
    kv_indices=torch.empty(0, dtype=torch.int64),
    offset=0,
    request_id="request",
)
"""
    env = os.environ.copy()
    env["LMCACHE_LOG_LEVEL"] = "INFO"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert deprecation.IN_PROCESS_DEPRECATION_MESSAGE not in (
        result.stdout + result.stderr
    )
