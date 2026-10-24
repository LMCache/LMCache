# SPDX-License-Identifier: Apache-2.0

"""Tests for the eviction loop's opt-in L1 memory usage logging."""

# Standard
from types import SimpleNamespace
from unittest.mock import Mock
import argparse
import logging
import threading
import time

# Third Party
import pytest

# First Party
from lmcache import torch_dev
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    add_storage_manager_args,
    parse_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.storage_controllers import eviction_controller
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
    L2AdapterEvictionState,
    L2EvictionController,
)
from lmcache.v1.mp_observability.config import add_observability_args

_EVICTION_LOGGER = "lmcache.v1.distributed.storage_controllers.eviction_controller"


def test_l2_failure_logs_are_throttled_per_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persistent failures stay visible without flooding or blocking peers."""
    ticks = iter([0.0, 1.0, 59.0, 60.0])
    now = 0.0
    passes = 0
    done = threading.Event()

    def advance_clock(_: float) -> None:
        nonlocal now
        now = next(ticks, now)
        time.sleep(0.001)

    def healthy_usage() -> SimpleNamespace:
        nonlocal passes
        passes += 1
        if passes >= 4:
            done.set()
        return SimpleNamespace(usage_fraction=0.0)

    monkeypatch.setattr(
        eviction_controller,
        "time",
        SimpleNamespace(monotonic=lambda: now, sleep=advance_clock),
    )
    broken = [Mock(spec=L2AdapterInterface), Mock(spec=L2AdapterInterface)]
    for adapter in broken:
        adapter.get_usage.side_effect = RuntimeError("adapter unavailable")
    healthy = Mock(spec=L2AdapterInterface)
    healthy.get_usage.side_effect = healthy_usage
    states = [
        L2AdapterEvictionState(i, adapter, EvictionConfig(eviction_policy="LRU"))
        for i, adapter in enumerate([*broken, healthy])
    ]
    controller = L2EvictionController(states)
    handler = _CaptureHandler()
    logger = logging.getLogger(_EVICTION_LOGGER)
    logger.addHandler(handler)
    try:
        controller.start()
        assert done.wait(timeout=5)
    finally:
        controller.stop()
        logger.removeHandler(handler)
    failures = [record for record in handler.records if record.exc_info]
    assert len(failures) == 4
    for adapter_id in (0, 1):
        assert sum(f"adapter_id={adapter_id};" in r.getMessage() for r in failures) == 2
    assert healthy.get_usage.call_count >= 4


class TestExtraLoggingConfigPlumbing:
    def test_composed_parser_populates_eviction_config(self):
        parser = argparse.ArgumentParser()
        add_storage_manager_args(parser)
        add_observability_args(parser)
        args = parser.parse_args(
            [
                "--l1-size-gb",
                "1",
                "--eviction-policy",
                "noop",
                "--enable-extra-logging",
                "--extra-logging-interval",
                "3.0",
            ]
        )
        config = parse_args_to_config(args)
        assert config.eviction_config.extra_logging_enabled is True
        assert config.eviction_config.extra_logging_interval == 3.0

    def test_standalone_parser_defaults_to_disabled(self):
        config = parse_args(["--l1-size-gb", "1", "--eviction-policy", "noop"])
        assert config.eviction_config.extra_logging_enabled is False
        assert config.eviction_config.extra_logging_interval == 10.0


@pytest.fixture
def l1_manager():
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=128 * 1024 * 1024,
            use_lazy=torch_dev.is_available(),
            init_size_in_bytes=64 * 1024 * 1024,
            align_bytes=0x1000,
        ),
        write_ttl_seconds=600,
        read_ttl_seconds=300,
    )
    mgr = L1Manager(config)
    yield mgr
    mgr.close()


class _CaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run_controller_and_capture(
    l1_manager: L1Manager, eviction_config: EvictionConfig
) -> list[str]:
    handler = _CaptureHandler()
    lg = logging.getLogger(_EVICTION_LOGGER)
    old_level = lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.INFO)
    controller = L1EvictionController(
        l1_manager=l1_manager, eviction_config=eviction_config
    )
    try:
        controller.start()
        time.sleep(1.3)
    finally:
        controller.stop()
        lg.removeHandler(handler)
        lg.setLevel(old_level)
    return [r.getMessage() for r in handler.records if r.levelno == logging.INFO]


class TestEvictionLoopMemoryLogging:
    def test_logs_memory_usage_when_enabled(self, l1_manager):
        messages = _run_controller_and_capture(
            l1_manager,
            EvictionConfig(
                eviction_policy="noop",
                extra_logging_enabled=True,
                extra_logging_interval=0.001,
            ),
        )
        used_bytes, total_bytes = l1_manager.get_memory_usage()
        pct = 0.0 if total_bytes == 0 else used_bytes / total_bytes * 100.0
        expected = (
            f"L1 memory usage: {used_bytes / (1 << 30):.2f}"
            f"/{total_bytes / (1 << 30):.2f} GiB ({pct:.1f}%)"
        )
        assert any(expected in m for m in messages)

    def test_no_memory_usage_log_by_default(self, l1_manager):
        messages = _run_controller_and_capture(
            l1_manager, EvictionConfig(eviction_policy="noop")
        )
        assert not any("L1 memory usage" in m for m in messages)
