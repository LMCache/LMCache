# SPDX-License-Identifier: Apache-2.0
"""Registration loss must trigger recovery even without a server outage."""

# Standard
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.client.heartbeat import HeartbeatThread


@pytest.mark.parametrize("first_recovery", [True, False])
def test_missing_registration_recovers_before_publishing_health(
    first_recovery: bool,
) -> None:
    health = threading.Event()
    health.set()
    recovered = threading.Event()
    calls: list[bool] = []
    heartbeat = HeartbeatThread(MagicMock(), health, interval=0.01)
    heartbeat.register_probe_callback(recovered.is_set)

    def recover() -> bool:
        calls.append(health.is_set())
        return first_recovery or len(calls) == 2

    def publish() -> bool:
        health.set()
        recovered.set()
        return True

    heartbeat.register_recover_callback(recover)
    heartbeat.register_healthy_callback(publish)
    try:
        heartbeat.start()
        assert recovered.wait(5)
    finally:
        heartbeat.stop_and_wait()
    assert calls == [False] * (1 if first_recovery else 2)
    assert health.is_set()


def test_unreachable_server_does_not_attempt_registration() -> None:
    health = threading.Event()
    health.set()
    degraded = threading.Event()
    recover = MagicMock(return_value=True)
    heartbeat = HeartbeatThread(MagicMock(), health, interval=0.01)

    def probe() -> None:
        return None

    heartbeat.register_probe_callback(probe)
    heartbeat.register_recover_callback(recover)
    heartbeat.register_unhealthy_callback(degraded.set)
    try:
        heartbeat.start()
        assert degraded.wait(5)
    finally:
        heartbeat.stop_and_wait()
    recover.assert_not_called()
    assert not health.is_set()
