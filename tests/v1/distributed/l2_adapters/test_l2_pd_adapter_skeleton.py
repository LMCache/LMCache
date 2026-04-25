# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PdL2Adapter skeleton (PR 2/7).

Tests are written against the public interface and docstring contract
of PdL2Adapter.  No implementation internals are accessed except for
the stub members that are explicitly part of the contract for this PR.
"""

# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.pd_l2_adapter import (
    PdL2Adapter,
    PdL2AdapterConfig,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sender_config():
    """Minimal sender config."""
    return PdL2AdapterConfig(
        role="sender",
        peer_host="192.168.1.10",
        peer_init_port=[9000],
        peer_alloc_port=[9001],
    )


@pytest.fixture
def receiver_config():
    """Minimal receiver config."""
    return PdL2AdapterConfig(
        role="receiver",
        peer_host="192.168.1.20",
        peer_init_port=[9100, 9101],
        peer_alloc_port=[9200, 9201],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_instantiate_sender(sender_config):
    """Create a PdL2Adapter with role=sender; no crash."""
    adapter = PdL2Adapter(sender_config)
    assert adapter._role == "sender"
    adapter.close()


def test_instantiate_receiver(receiver_config):
    """Create a PdL2Adapter with role=receiver; no crash."""
    adapter = PdL2Adapter(receiver_config)
    assert adapter._role == "receiver"
    adapter.close()


def test_eventfd_getters_return_valid_fd(sender_config):
    """All 3 eventfd getters return integer file descriptors > 0."""
    adapter = PdL2Adapter(sender_config)
    try:
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        assert isinstance(store_fd, int) and store_fd > 0
        assert isinstance(lookup_fd, int) and lookup_fd > 0
        assert isinstance(load_fd, int) and load_fd > 0
    finally:
        adapter.close()


def test_eventfd_unique(sender_config):
    """All 3 eventfds are distinct values."""
    adapter = PdL2Adapter(sender_config)
    try:
        store_fd = adapter.get_store_event_fd()
        lookup_fd = adapter.get_lookup_and_lock_event_fd()
        load_fd = adapter.get_load_event_fd()

        # All three must be different
        assert store_fd != lookup_fd
        assert store_fd != load_fd
        assert lookup_fd != load_fd
    finally:
        adapter.close()


def test_close_releases_fds(sender_config):
    """After close(), eventfds are invalid (os.write raises)."""
    adapter = PdL2Adapter(sender_config)
    store_fd = adapter.get_store_event_fd()
    lookup_fd = adapter.get_lookup_and_lock_event_fd()
    load_fd = adapter.get_load_event_fd()

    adapter.close()

    # Attempting to write to closed fds should raise OSError
    with pytest.raises(OSError):
        os.eventfd_write(store_fd, 1)
    with pytest.raises(OSError):
        os.eventfd_write(lookup_fd, 1)
    with pytest.raises(OSError):
        os.eventfd_write(load_fd, 1)


def test_close_is_idempotent(sender_config):
    """Calling close() multiple times is safe and does not raise."""
    adapter = PdL2Adapter(sender_config)
    adapter.close()
    adapter.close()  # Should not raise or cause issues


def test_stub_methods_raise(sender_config):
    """Each submit/query method raises NotImplementedError."""
    adapter = PdL2Adapter(sender_config)
    try:
        # Store interface
        with pytest.raises(NotImplementedError, match="impl in PR 4/7"):
            adapter.submit_store_task([], [])
        with pytest.raises(NotImplementedError, match="impl in PR 4/7"):
            adapter.pop_completed_store_tasks()

        # Lookup and lock interface
        with pytest.raises(NotImplementedError, match="impl in PR 4/7"):
            adapter.submit_lookup_and_lock_task([])
        with pytest.raises(NotImplementedError, match="impl in PR 4/7"):
            adapter.query_lookup_and_lock_result(0)
        with pytest.raises(NotImplementedError, match="impl in PR 4/7"):
            adapter.submit_unlock([])

        # Load interface
        with pytest.raises(NotImplementedError, match="impl in PR 4/7"):
            adapter.submit_load_task([], [])
        with pytest.raises(NotImplementedError, match="impl in PR 4/7"):
            adapter.query_load_result(0)
    finally:
        adapter.close()


def test_stop_flag_set_on_close(sender_config):
    """_stop_flag.is_set() returns True after close()."""
    adapter = PdL2Adapter(sender_config)
    assert not adapter._stop_flag.is_set()
    adapter.close()
    assert adapter._stop_flag.is_set()


def test_factory_creates_instance(sender_config):
    """Factory('pd', config) returns a PdL2Adapter instance."""
    adapter = create_l2_adapter_from_registry(sender_config)
    assert isinstance(adapter, PdL2Adapter)
    adapter.close()
