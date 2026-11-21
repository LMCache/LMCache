# SPDX-License-Identifier: Apache-2.0
"""Tests for L2HealthMonitor.

The monitor decides, from the probe results of L2 adapters, which ones may take
new work. These tests drive it with adapters whose probe results and reconnect
behavior the test controls, through the public interface only: ``check_once``,
``is_available`` and ``report_status``.
"""

# Standard
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import L2HealthConfig
from lmcache.v1.distributed.l2_adapters.health import (
    HealthProbe,
    L2ReconnectError,
    ProbeStatus,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
    MockL2Adapter,
    MockL2AdapterConfig,
)
from lmcache.v1.distributed.l2_health_monitor import L2HealthMonitor

OK = HealthProbe(ProbeStatus.OK)
DOWN = HealthProbe(ProbeStatus.UNREACHABLE, "backend down")
BROKEN = HealthProbe(ProbeStatus.RECONNECT_NEEDED, "connection broken")


class _ProbedAdapter(MockL2Adapter):
    """A mock adapter whose probe results and reconnect the test scripts."""

    def __init__(
        self,
        results: list[HealthProbe],
        *,
        probe_supported: bool = True,
        reconnect_supported: bool = False,
    ) -> None:
        super().__init__(MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1))
        self._results = list(results)
        self._probe_supported = probe_supported
        self._reconnect_supported = reconnect_supported
        self.probe_calls = 0
        self.reconnect_calls = 0
        # When set, the next probe blocks until the event is set.
        self.hold_probe: threading.Event | None = None
        # When set, reconnect raises it.
        self.reconnect_error: Exception | None = None

    def push_result(self, result: HealthProbe) -> None:
        """Make every following probe return ``result``."""
        self._results = [result]

    @property
    def supports_health_probe(self) -> bool:
        return self._probe_supported

    def probe_health(self, timeout_s: float) -> HealthProbe:
        self.probe_calls += 1
        if self.hold_probe is not None:
            self.hold_probe.wait()
        return self._results.pop(0) if len(self._results) > 1 else self._results[0]

    @property
    def supports_reconnect(self) -> bool:
        return self._reconnect_supported

    def reconnect(self) -> None:
        self.reconnect_calls += 1
        if self.reconnect_error is not None:
            raise self.reconnect_error


def _monitor(threshold: int = 2, probe_timeout_s: float = 0.3) -> L2HealthMonitor:
    return L2HealthMonitor(
        L2HealthConfig(
            check_interval_s=0,
            probe_timeout_s=probe_timeout_s,
            failure_threshold=threshold,
        )
    )


def _state(monitor: L2HealthMonitor, adapter_id: int = 0) -> str:
    adapters = monitor.report_status()["adapters"]
    return next(a["state"] for a in adapters if a["adapter_id"] == adapter_id)


def _wait_for_state(
    monitor: L2HealthMonitor, wanted: str, adapter_id: int = 0, timeout_s: float = 5.0
) -> None:
    deadline = time.monotonic() + timeout_s
    while _state(monitor, adapter_id) != wanted:
        assert time.monotonic() < deadline, (
            f"adapter {adapter_id} still {_state(monitor, adapter_id)}, wanted {wanted}"
        )
        time.sleep(0.02)


@pytest.fixture
def adapters():
    created: list[_ProbedAdapter] = []

    def make(*args, **kwargs) -> _ProbedAdapter:
        adapter = _ProbedAdapter(*args, **kwargs)
        created.append(adapter)
        return adapter

    yield make
    for adapter in created:
        adapter.close()


class TestAdaptersWithoutProbeSupport:
    def test_never_probed_and_always_available(self, adapters):
        adapter = adapters([DOWN], probe_supported=False)
        monitor = _monitor()
        monitor.register(0, adapter, "mock")

        for _ in range(5):
            monitor.check_once()

        assert adapter.probe_calls == 0
        assert monitor.is_available(0)
        assert monitor.report_status()["adapters"] == []
        assert not monitor.report_status()["degraded"]

    def test_unknown_adapter_is_available(self):
        assert _monitor().is_available(12345)


class TestFailureDetection:
    def test_unavailable_only_after_threshold_consecutive_failures(self, adapters):
        adapter = adapters([DOWN])
        monitor = _monitor(threshold=3)
        monitor.register(0, adapter, "mock")

        monitor.check_once()
        monitor.check_once()
        assert monitor.is_available(0)
        monitor.check_once()
        assert not monitor.is_available(0)
        assert _state(monitor) == "unavailable"
        status = monitor.report_status()
        assert status["degraded"]
        assert status["adapters"][0]["reason"] == "backend down"

    def test_a_success_resets_the_failure_count(self, adapters):
        adapter = adapters([DOWN, DOWN, OK, DOWN, DOWN, OK])
        monitor = _monitor(threshold=3)
        monitor.register(0, adapter, "mock")

        for _ in range(6):
            monitor.check_once()

        assert monitor.is_available(0)
        assert monitor.report_status()["adapters"][0]["consecutive_failures"] == 0

    def test_available_again_after_one_successful_probe(self, adapters):
        adapter = adapters([DOWN])
        monitor = _monitor(threshold=1)
        monitor.register(0, adapter, "mock")
        monitor.check_once()
        assert not monitor.is_available(0)

        adapter.push_result(OK)
        monitor.check_once()

        assert monitor.is_available(0)
        assert not monitor.report_status()["degraded"]

    def test_a_probe_that_raises_counts_as_a_failure(self, adapters):
        adapter = adapters([OK])
        monitor = _monitor(threshold=1)
        monitor.register(0, adapter, "mock")

        def boom(timeout_s: float) -> HealthProbe:
            raise RuntimeError("probe bug")

        adapter.probe_health = boom  # type: ignore[method-assign]
        monitor.check_once()

        assert not monitor.is_available(0)
        assert "probe bug" in monitor.report_status()["adapters"][0]["reason"]

    def test_a_hung_probe_counts_as_a_failure_without_blocking_others(self, adapters):
        hung = adapters([OK])
        hung.hold_probe = threading.Event()
        healthy = adapters([OK])
        monitor = _monitor(threshold=1, probe_timeout_s=0.2)
        monitor.register(0, hung, "hung")
        monitor.register(1, healthy, "healthy")

        started = time.monotonic()
        monitor.check_once()
        elapsed = time.monotonic() - started

        try:
            assert elapsed < 3.0
            assert not monitor.is_available(0)
            assert monitor.is_available(1)
            assert healthy.probe_calls == 1

            # While the first probe is still stuck, no second one piles up.
            monitor.check_once()
            assert hung.probe_calls == 1
        finally:
            hung.hold_probe.set()

    def test_unregistered_adapter_is_available_again(self, adapters):
        adapter = adapters([DOWN])
        monitor = _monitor(threshold=1)
        monitor.register(0, adapter, "mock")
        monitor.check_once()
        assert not monitor.is_available(0)

        monitor.unregister(0)

        assert monitor.is_available(0)


class TestReconnect:
    def test_reconnects_immediately_once_the_backend_is_back(self, adapters):
        adapter = adapters([DOWN], reconnect_supported=True)
        monitor = _monitor(threshold=1)
        monitor.register(0, adapter, "mock")
        monitor.check_once()
        assert _state(monitor) == "unavailable"

        adapter.push_result(BROKEN)
        monitor.check_once()

        _wait_for_state(monitor, "healthy")
        assert adapter.reconnect_calls == 1
        assert monitor.is_available(0)

    def test_a_healthy_adapter_with_a_broken_connection_needs_the_threshold(
        self, adapters
    ):
        adapter = adapters([BROKEN], reconnect_supported=True)
        monitor = _monitor(threshold=2)
        monitor.register(0, adapter, "mock")

        monitor.check_once()
        assert adapter.reconnect_calls == 0
        assert monitor.is_available(0)

        monitor.check_once()
        _wait_for_state(monitor, "healthy")
        assert adapter.reconnect_calls == 1

    def test_not_available_while_reconnecting(self, adapters):
        adapter = adapters([BROKEN], reconnect_supported=True)
        gate = threading.Event()
        original = adapter.reconnect

        def slow_reconnect() -> None:
            gate.wait()
            original()

        adapter.reconnect = slow_reconnect  # type: ignore[method-assign]
        monitor = _monitor(threshold=1)
        monitor.register(0, adapter, "mock")

        monitor.check_once()
        try:
            assert _state(monitor) == "recovering"
            assert not monitor.is_available(0)
            # Nothing is probed while the reconnect runs.
            monitor.check_once()
            assert adapter.probe_calls == 1
        finally:
            gate.set()
        _wait_for_state(monitor, "healthy")

    def test_failed_reconnect_leaves_it_unavailable_and_is_retried(self, adapters):
        adapter = adapters([BROKEN], reconnect_supported=True)
        adapter.reconnect_error = L2ReconnectError("still down")
        monitor = _monitor(threshold=1)
        monitor.register(0, adapter, "mock")

        monitor.check_once()
        _wait_for_state(monitor, "unavailable")
        assert "still down" in monitor.report_status()["adapters"][0]["reason"]

        adapter.reconnect_error = None
        monitor.check_once()

        _wait_for_state(monitor, "healthy")
        assert adapter.reconnect_calls == 2

    def test_unexpected_reconnect_error_leaves_it_unavailable(self, adapters):
        adapter = adapters([BROKEN], reconnect_supported=True)
        adapter.reconnect_error = RuntimeError("adapter bug")
        monitor = _monitor(threshold=1)
        monitor.register(0, adapter, "mock")

        monitor.check_once()

        _wait_for_state(monitor, "unavailable")

    def test_without_reconnect_support_a_broken_connection_is_a_plain_failure(
        self, adapters
    ):
        adapter = adapters([BROKEN], reconnect_supported=False)
        monitor = _monitor(threshold=2)
        monitor.register(0, adapter, "mock")

        monitor.check_once()
        monitor.check_once()

        assert adapter.reconnect_calls == 0
        assert not monitor.is_available(0)


class TestBackgroundLoop:
    def test_loop_detects_failure_and_recovery(self, adapters):
        adapter = adapters([DOWN])
        monitor = L2HealthMonitor(
            L2HealthConfig(
                check_interval_s=0.05, probe_timeout_s=0.3, failure_threshold=2
            )
        )
        monitor.register(0, adapter, "mock")
        monitor.start()
        try:
            _wait_for_state(monitor, "unavailable")
            adapter.push_result(OK)
            _wait_for_state(monitor, "healthy")
        finally:
            monitor.stop()

    def test_interval_zero_disables_probing(self, adapters):
        adapter = adapters([DOWN])
        monitor = L2HealthMonitor(
            L2HealthConfig(check_interval_s=0, probe_timeout_s=0.3, failure_threshold=1)
        )
        monitor.register(0, adapter, "mock")

        monitor.start()
        time.sleep(0.3)
        monitor.stop()

        assert adapter.probe_calls == 0
        assert monitor.is_available(0)
        assert not monitor.report_status()["enabled"]


class TestStatusReport:
    def test_reports_each_managed_adapter(self, adapters):
        down = adapters([DOWN])
        up = adapters([OK])
        monitor = _monitor(threshold=1)
        monitor.register(3, down, "s3")
        monitor.register(5, up, "resp")

        monitor.check_once()

        status = monitor.report_status()
        by_id = {a["adapter_id"]: a for a in status["adapters"]}
        assert by_id[3]["l2_name"] == "s3"
        assert by_id[3]["state"] == "unavailable"
        assert by_id[5]["state"] == "healthy"
        assert by_id[5]["seconds_since_last_ok"] is not None
        assert status["degraded"]
        observations = dict(
            (attrs["adapter_index"], value)
            for value, attrs in monitor.get_availability_observations()
        )
        assert observations == {3: 0, 5: 1}


class TestConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"check_interval_s": -1},
            {"probe_timeout_s": 0},
            {"failure_threshold": 0},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            L2HealthConfig(**kwargs)
