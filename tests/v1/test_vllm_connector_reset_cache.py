# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM connector cache reset integration."""

# Standard
from enum import Enum
from importlib import import_module
from types import SimpleNamespace
from typing import Any, Iterator, cast
import logging
import sys
import types

# Third Party
import pytest

_TESTED_MODULES = (
    "lmcache.integration.vllm.lmcache_connector_v1",
    "lmcache.integration.vllm.vllm_v1_adapter",
)


class _KVConnectorRole(Enum):
    SCHEDULER = "scheduler"
    WORKER = "worker"


@pytest.fixture
def connector_modules(monkeypatch) -> Iterator[SimpleNamespace]:
    """Load connector modules with the minimal vLLM API they import."""
    for module_name in _TESTED_MODULES:
        sys.modules.pop(module_name, None)

    _install_vllm_stubs(monkeypatch)

    dynamic_module = import_module("lmcache.integration.vllm.lmcache_connector_v1")
    adapter_module = import_module("lmcache.integration.vllm.vllm_v1_adapter")

    yield SimpleNamespace(
        KVConnectorRole=_KVConnectorRole,
        LMCacheConnectorV1Dynamic=dynamic_module.LMCacheConnectorV1Dynamic,
        LMCacheConnectorV1Impl=adapter_module.LMCacheConnectorV1Impl,
    )

    for module_name in _TESTED_MODULES:
        sys.modules.pop(module_name, None)


def _install_vllm_stubs(monkeypatch) -> None:
    module_names = [
        "vllm",
        "vllm.config",
        "vllm.distributed",
        "vllm.distributed.kv_transfer",
        "vllm.distributed.kv_transfer.kv_connector",
        "vllm.distributed.kv_transfer.kv_connector.v1",
        "vllm.distributed.kv_transfer.kv_connector.v1.base",
        "vllm.distributed.parallel_state",
        "vllm.logger",
        "vllm.sampling_params",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm.v1.core.sched.output",
        "vllm.v1.request",
        "vllm.version",
    ]
    for module_name in module_names:
        module = types.ModuleType(module_name)
        if module_name in {
            "vllm",
            "vllm.distributed",
            "vllm.distributed.kv_transfer",
            "vllm.distributed.kv_transfer.kv_connector",
            "vllm.distributed.kv_transfer.kv_connector.v1",
            "vllm.v1",
            "vllm.v1.core",
            "vllm.v1.core.sched",
        }:
            module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, module)

    config_module = cast(Any, sys.modules["vllm.config"])
    config_module.VllmConfig = type("VllmConfig", (), {})

    base_module = cast(
        Any,
        sys.modules["vllm.distributed.kv_transfer.kv_connector.v1.base"],
    )
    base_module.KVConnectorBase_V1 = type("KVConnectorBase_V1", (), {})
    base_module.KVConnectorMetadata = type("KVConnectorMetadata", (), {})
    base_module.KVConnectorRole = _KVConnectorRole

    parallel_state_module = cast(Any, sys.modules["vllm.distributed.parallel_state"])
    parallel_state_module.get_pp_group = lambda: None

    logger_module = cast(Any, sys.modules["vllm.logger"])
    logger_module.init_logger = logging.getLogger

    sampling_params_module = cast(Any, sys.modules["vllm.sampling_params"])
    sampling_params_module.SamplingParams = type("SamplingParams", (), {})

    sched_output_module = cast(Any, sys.modules["vllm.v1.core.sched.output"])
    sched_output_module.SchedulerOutput = type("SchedulerOutput", (), {})

    request_module = cast(Any, sys.modules["vllm.v1.request"])
    request_module.RequestStatus = SimpleNamespace(FINISHED_ABORTED="aborted")

    version_module = cast(Any, sys.modules["vllm.version"])
    version_module.__version__ = "test"


class _FakeEngine:
    def __init__(self, clear_return: int = 128) -> None:
        self.clear_return = clear_return
        self.clear_calls = 0

    def clear(self) -> int:
        self.clear_calls += 1
        return self.clear_return


class _FakeLookupClient:
    def __init__(self) -> None:
        self.cancelled: list[str] = []
        self.cleared: list[str] = []

    def cancel_lookup(self, lookup_id: str) -> None:
        self.cancelled.append(lookup_id)

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.cleared.append(lookup_id)


def _make_impl(
    connector_modules: SimpleNamespace,
    role: _KVConnectorRole,
    *,
    engine: object = None,
    lookup_client: object = None,
) -> Any:
    connector_cls = connector_modules.LMCacheConnectorV1Impl
    connector = connector_cls.__new__(connector_cls)
    connector._role = role
    connector._manager = SimpleNamespace(
        lmcache_engine=engine,
        lookup_client=lookup_client,
    )
    connector.load_specs = {"req-load": object()}
    connector._unfinished_requests = {"req-running": object()}
    connector._request_trackers = {"req-tracker": object()}
    connector._requests_priority = {"req-priority": 1}
    connector._invalid_block_ids = {7}
    connector.layerwise_retrievers = [object()]
    connector._layerwise_save_storers = {"req-storer": object()}
    return connector


def test_dynamic_connector_reset_cache_delegates_to_impl(
    connector_modules: SimpleNamespace,
) -> None:
    """vLLM's public connector hook delegates to the LMCache adapter."""
    connector_cls = connector_modules.LMCacheConnectorV1Dynamic
    connector = connector_cls.__new__(connector_cls)
    connector._lmcache_engine = SimpleNamespace(reset_cache=lambda: True)

    assert connector.reset_cache() is True


def test_scheduler_reset_cache_clears_engine_and_local_state(
    connector_modules: SimpleNamespace,
) -> None:
    """Scheduler reset clears LMCache and drops request-scoped state."""
    adapter_module = import_module("lmcache.integration.vllm.vllm_v1_adapter")
    adapter_module.tmp_disagg_tracker["req-running"] = object()
    engine = _FakeEngine()
    lookup_client = _FakeLookupClient()
    connector = _make_impl(
        connector_modules,
        connector_modules.KVConnectorRole.SCHEDULER,
        engine=engine,
        lookup_client=lookup_client,
    )

    assert connector.reset_cache() is True

    assert engine.clear_calls == 1
    assert connector.load_specs == {}
    assert connector._unfinished_requests == {}
    assert connector._request_trackers == {}
    assert connector._requests_priority == {}
    assert connector._invalid_block_ids == set()
    assert connector.layerwise_retrievers == []
    assert connector._layerwise_save_storers == {}
    assert "req-running" not in adapter_module.tmp_disagg_tracker
    expected_lookup_ids = {"req-load", "req-priority", "req-running", "req-tracker"}
    assert set(lookup_client.cancelled) == expected_lookup_ids
    assert set(lookup_client.cleared) == expected_lookup_ids


def test_scheduler_reset_cache_returns_false_without_engine(
    connector_modules: SimpleNamespace,
) -> None:
    """A scheduler without an LMCache engine must not report reset success."""
    connector = _make_impl(
        connector_modules,
        connector_modules.KVConnectorRole.SCHEDULER,
    )

    assert connector.reset_cache() is False


def test_worker_reset_cache_is_noop(connector_modules: SimpleNamespace) -> None:
    """Worker role does not own vLLM's scheduler-side reset hook."""
    engine = _FakeEngine()
    connector = _make_impl(
        connector_modules,
        connector_modules.KVConnectorRole.WORKER,
        engine=engine,
    )

    assert connector.reset_cache() is None
    assert engine.clear_calls == 0
