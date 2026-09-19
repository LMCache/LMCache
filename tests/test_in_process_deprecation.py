# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the in-process runtime deprecation warning."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Iterator
import logging
import os
import subprocess
import sys

# Third Party
import pytest
import torch

# First Party
from lmcache import deprecation

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_engine import LMCacheEngine

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


def _construct_legacy_engine(monkeypatch: pytest.MonkeyPatch) -> "LMCacheEngine":
    # First Party
    from lmcache.v1 import cache_engine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.token_database import ChunkedTokenDatabase

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

    config = LMCacheEngineConfig.from_defaults(py_enable_gc=True)
    metadata = LMCacheMetadata(
        model_name="test",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.float16,
        kv_shape=(1, 2, config.chunk_size, 1, 1),
        role="worker",
    )
    return cache_engine.LMCacheEngine(
        config,
        metadata,
        ChunkedTokenDatabase(config, metadata),
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
    assert _deprecation_messages(deprecation_records) == [
        deprecation.IN_PROCESS_DEPRECATION_MESSAGE
    ]
    _construct_legacy_engine(monkeypatch)

    assert _deprecation_messages(deprecation_records) == [
        deprecation.IN_PROCESS_DEPRECATION_MESSAGE
    ]


def _run_python(script: str, *python_args: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(LMCACHE_LOG_LEVEL="INFO", LMCACHE_TRACK_USAGE="false")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repo_root), env.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, *python_args, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _assert_silent(script: str) -> None:
    # Prove capture is working, without consuming the once-per-process guard
    # until after the imports/initialization under test.
    result = _run_python(
        script
        + """
import sys
from lmcache.deprecation import warn_in_process_mode_deprecated
print("DEPRECATION_CAPTURE_CONTROL", file=sys.stderr, flush=True)
warn_in_process_mode_deprecated()
"""
    )
    before, marker, after = result.stderr.partition("DEPRECATION_CAPTURE_CONTROL")
    assert marker
    assert deprecation.IN_PROCESS_DEPRECATION_MESSAGE not in result.stdout + before
    assert after.count(deprecation.IN_PROCESS_DEPRECATION_MESSAGE) == 1


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork")
def test_forked_child_emits_its_own_warning() -> None:
    # Fork a fresh interpreter, never pytest's potentially multithreaded process.
    result = _run_python(
        """
import os
import signal
import sys
import types
import warnings

# Load the real helper and logging modules without accelerator discovery in
# lmcache.__init__, which starts native threads even when CUDA is hidden.
package = types.ModuleType("lmcache")
package.__path__ = [os.path.join(os.getcwd(), "lmcache")]
sys.modules["lmcache"] = package
from lmcache.deprecation import warn_in_process_mode_deprecated

warn_in_process_mode_deprecated()
warn_in_process_mode_deprecated()
warnings.filterwarnings("error", message=".*multi-threaded.*",
                        category=DeprecationWarning)
if sys.platform.startswith("linux"):
    assert len(os.listdir("/proc/self/task")) == 1
child_pid = os.fork()
if child_pid == 0:
    signal.alarm(15)
    try:
        warn_in_process_mode_deprecated()
        warn_in_process_mode_deprecated()
    except BaseException:
        os._exit(1)
    os._exit(0)
_, status = os.waitpid(child_pid, 0)
assert os.waitstatus_to_exitcode(status) == 0
""",
        "-S",
    )
    assert result.stderr.count(deprecation.IN_PROCESS_DEPRECATION_MESSAGE) == 2


def test_shared_config_and_metadata_are_silent() -> None:
    _assert_silent("""
import torch
from lmcache.integration.sglang.sglang_adapter import LoadMetadata, StoreMetadata
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
""")


@pytest.mark.parametrize(
    "module",
    [
        "lmcache.integration.atom.multi_process_adapter",
        "lmcache.integration.sglang.multi_process_adapter",
        "lmcache.integration.vllm.vllm_multi_process_adapter",
        "lmcache.integration.vllm.vllm_v1_adapter",
    ],
)
def test_adapter_import_is_silent(module: str) -> None:
    if module.endswith("vllm_v1_adapter"):
        pytest.importorskip("vllm")
    _assert_silent(f"import {module}\n")


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_mp_server_startup_is_silent(transport: str) -> None:
    _assert_silent(f"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from lmcache.v1.distributed.config import (
    EvictionConfig, L1ManagerConfig, L1MemoryManagerConfig, StorageManagerConfig,
)
from lmcache.v1.mp_observability.config import ObservabilityConfig
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.server import run_cache_server

server, engine = run_cache_server(
    MPServerConfig(host="127.0.0.1", port=0, shm_name="", transport={transport!r}),
    StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(L1MemoryManagerConfig(
            size_in_bytes=1 << 20, use_lazy=False, shm_name="",
        )),
        eviction_config=EvictionConfig(eviction_policy="LRU"),
    ),
    ObservabilityConfig(enabled=False),
    return_engine=True,
    start_prometheus_http_server=False,
)
try:
    assert engine.report_status()["is_healthy"]
finally:
    server.close()
    engine.close()
""")


def test_mp_scheduler_startup_is_silent(
    monkeypatch: pytest.MonkeyPatch,
    deprecation_records: list[logging.LogRecord],
) -> None:
    # Third Party
    import zmq

    # First Party
    from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_module
    from lmcache.v1.multiprocess.futures import MessagingFuture

    chunk_size: MessagingFuture[int] = MessagingFuture()
    chunk_size.set_result(256)
    client = SimpleNamespace(get_chunk_size=lambda: chunk_size, close=lambda: None)
    monkeypatch.setattr(
        adapter_module.RequestClientFactory,
        "create",
        lambda *_args, **_kwargs: client,
    )
    with zmq.Context() as context:
        adapter = adapter_module.LMCacheMPSchedulerAdapter(
            ["tcp://test-server"],
            context,
            "test",
            16,
            adapter_module.ParallelStrategy(False, 1, 0, 1, 1, 1),
        )
        try:
            assert adapter.is_healthy
            assert adapter.blocks_in_chunk == 16
        finally:
            adapter.shutdown()

    assert not _deprecation_messages(deprecation_records)
    deprecation.warn_in_process_mode_deprecated()
    assert _deprecation_messages(deprecation_records) == [
        deprecation.IN_PROCESS_DEPRECATION_MESSAGE
    ]
