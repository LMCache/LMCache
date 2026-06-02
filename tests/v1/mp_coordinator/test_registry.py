# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mp coordinator instance registry."""

# Standard
from unittest.mock import Mock
import time

# First Party
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstanceNode


def _node(instance_id: str, heartbeat: float = 0.0) -> MPInstanceNode:
    """Build an MPInstanceNode with a mock command socket for tests."""
    now = heartbeat or time.time()
    return MPInstanceNode(
        instance_id=instance_id,
        ip="127.0.0.1",
        control_port=5000,
        command_socket=Mock(),
        registration_time=now,
        last_heartbeat_time=now,
    )


def test_register_and_get():
    registry = InstanceRegistry()
    node = _node("a")
    registry.register(node)
    assert registry.contains("a")
    assert registry.get("a") is node
    assert registry.get("missing") is None


def test_deregister_returns_node():
    registry = InstanceRegistry()
    node = _node("a")
    registry.register(node)
    removed = registry.deregister("a")
    assert removed is node
    assert not registry.contains("a")
    assert registry.deregister("a") is None


def test_all_instances_snapshot_is_independent():
    registry = InstanceRegistry()
    registry.register(_node("a"))
    registry.register(_node("b"))
    snapshot = registry.all_instances()
    assert {n.instance_id for n in snapshot} == {"a", "b"}
    registry.deregister("a")
    # Snapshot taken earlier is unaffected by later mutation.
    assert {n.instance_id for n in snapshot} == {"a", "b"}


def test_update_heartbeat():
    registry = InstanceRegistry()
    node = _node("a", heartbeat=100.0)
    registry.register(node)
    assert registry.update_heartbeat("a", 200.0) is True
    assert registry.get("a").last_heartbeat_time == 200.0
    # Unknown instance signals a needed re-register.
    assert registry.update_heartbeat("missing", 300.0) is False


def test_stale_detects_expired():
    registry = InstanceRegistry()
    now = time.time()
    registry.register(_node("fresh", heartbeat=now))
    registry.register(_node("old", heartbeat=now - 100.0))
    stale = registry.stale(timeout=30.0)
    assert stale == ["old"]
