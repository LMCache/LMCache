# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for PdL2AdapterConfig and PdL2Adapter.

Tests are written against the public interface and docstring contract.
No implementation internals are accessed.
"""

# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.config import get_type_name_for_config
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.pd_l2_adapter import (
    PdL2Adapter,
    PdL2AdapterConfig,
)

# ---------------------------------------------------------------------------
# Minimal valid dicts
# ---------------------------------------------------------------------------

_MINIMAL_SENDER = {
    "role": "sender",
    "peer_host": "192.168.1.10",
    "peer_init_port": [9000],
    "peer_alloc_port": [9001],
}

_MINIMAL_RECEIVER = {
    "role": "receiver",
    "peer_host": "192.168.1.20",
    "peer_init_port": [9100, 9101],
    "peer_alloc_port": [9200, 9201],
}

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
# PdL2AdapterConfig tests
# ---------------------------------------------------------------------------


def test_parse_minimal_sender_config():
    """from_dict accepts a minimal sender dict and sets role='sender'."""
    cfg = PdL2AdapterConfig.from_dict(_MINIMAL_SENDER)
    assert cfg.role == "sender"
    assert cfg.peer_host == "192.168.1.10"
    assert cfg.peer_init_port == [9000]
    assert cfg.peer_alloc_port == [9001]


def test_parse_minimal_receiver_config():
    """from_dict accepts a minimal receiver dict and sets role='receiver'."""
    cfg = PdL2AdapterConfig.from_dict(_MINIMAL_RECEIVER)
    assert cfg.role == "receiver"
    assert cfg.peer_host == "192.168.1.20"
    assert cfg.peer_init_port == [9100, 9101]
    assert cfg.peer_alloc_port == [9200, 9201]


@pytest.mark.parametrize(
    "missing_key,match",
    [
        ("peer_host", "peer_host must be a non-empty string"),
        ("peer_init_port", "peer_init_port is required"),
        ("peer_alloc_port", "peer_alloc_port is required"),
    ],
)
def test_fail_on_missing_required(missing_key, match):
    """from_dict raises ValueError when a required field is absent."""
    d = dict(_MINIMAL_SENDER)
    del d[missing_key]
    with pytest.raises(ValueError, match=match):
        PdL2AdapterConfig.from_dict(d)


def test_fail_on_invalid_role():
    """from_dict raises ValueError when role is not 'sender' or 'receiver'."""
    d = dict(_MINIMAL_SENDER)
    d["role"] = "invalid"
    with pytest.raises(ValueError, match="role must be"):
        PdL2AdapterConfig.from_dict(d)


def test_help_contains_all_field_names():
    """help() mentions every config field name."""
    text = PdL2AdapterConfig.help()
    for field in (
        "role",
        "peer_host",
        "peer_init_port",
        "peer_alloc_port",
        "proxy_host",
        "proxy_port",
        "buffer_size",
        "buffer_device",
        "transfer_channel",
        "nixl_backends",
    ):
        assert field in text, f"help() is missing field {field!r}"


def test_registered_in_factory():
    """'pd' is registered in the L2 adapter type registry."""
    cfg = PdL2AdapterConfig.from_dict(_MINIMAL_SENDER)
    assert get_type_name_for_config(cfg) == "pd"


def test_all_fields_round_trip():
    """All fields passed to from_dict are stored correctly on the config."""
    d = {
        "role": "sender",
        "peer_host": "10.0.0.1",
        "peer_init_port": [8000, 8001],
        "peer_alloc_port": [8100, 8101],
        "proxy_host": "proxy.example.com",
        "proxy_port": 7000,
        "buffer_size": 134217728,
        "buffer_device": "cuda",
        "transfer_channel": "mock_memory",
        "nixl_backends": ["tcp", "rdma"],
    }
    cfg = PdL2AdapterConfig.from_dict(d)
    assert cfg.role == "sender"
    assert cfg.peer_host == "10.0.0.1"
    assert cfg.peer_init_port == [8000, 8001]
    assert cfg.peer_alloc_port == [8100, 8101]
    assert cfg.proxy_host == "proxy.example.com"
    assert cfg.proxy_port == 7000
    assert cfg.buffer_size == 134217728
    assert cfg.buffer_device == "cuda"
    assert cfg.transfer_channel == "mock_memory"
    assert cfg.nixl_backends == ["tcp", "rdma"]


def test_defaults_applied():
    """from_dict fills in correct default values for optional fields."""
    cfg = PdL2AdapterConfig.from_dict(_MINIMAL_SENDER)
    assert cfg.proxy_host == "127.0.0.1"
    assert cfg.proxy_port == 6688
    assert cfg.buffer_size == 1073741824
    assert cfg.buffer_device == "cpu"
    assert cfg.transfer_channel == "nixl"
    assert cfg.nixl_backends == ["tcp"]


def test_fail_on_invalid_buffer_device():
    """from_dict raises ValueError when buffer_device is not 'cpu' or 'cuda'."""
    d = dict(_MINIMAL_SENDER)
    d["buffer_device"] = "gpu"
    with pytest.raises(ValueError, match="buffer_device must be"):
        PdL2AdapterConfig.from_dict(d)


def test_fail_on_invalid_transfer_channel():
    """from_dict raises ValueError when transfer_channel is not allowed."""
    d = dict(_MINIMAL_SENDER)
    d["transfer_channel"] = "invalid"
    with pytest.raises(ValueError, match="transfer_channel must be"):
        PdL2AdapterConfig.from_dict(d)


# ---------------------------------------------------------------------------
# PdL2Adapter tests
# ---------------------------------------------------------------------------


def test_instantiate_sender(sender_config):
    """Create a PdL2Adapter with role=sender; no crash."""
    adapter = PdL2Adapter(sender_config)
    assert isinstance(adapter, PdL2Adapter)
    adapter.close()


def test_instantiate_receiver(receiver_config):
    """Create a PdL2Adapter with role=receiver; no crash."""
    adapter = PdL2Adapter(receiver_config)
    assert isinstance(adapter, PdL2Adapter)
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
        with pytest.raises(NotImplementedError, match="impl in following PRs"):
            adapter.submit_store_task([], [])
        with pytest.raises(NotImplementedError, match="impl in following PRs"):
            adapter.pop_completed_store_tasks()

        # Lookup and lock interface
        with pytest.raises(NotImplementedError, match="impl in following PRs"):
            adapter.submit_lookup_and_lock_task([])
        with pytest.raises(NotImplementedError, match="impl in following PRs"):
            adapter.query_lookup_and_lock_result(0)
        with pytest.raises(NotImplementedError, match="impl in following PRs"):
            adapter.submit_unlock([])

        # Load interface
        with pytest.raises(NotImplementedError, match="impl in following PRs"):
            adapter.submit_load_task([], [])
        with pytest.raises(NotImplementedError, match="impl in following PRs"):
            adapter.query_load_result(0)
    finally:
        adapter.close()


def test_factory_creates_instance(sender_config):
    """Factory('pd', config) returns a PdL2Adapter instance."""
    adapter = create_l2_adapter_from_registry(sender_config)
    assert isinstance(adapter, PdL2Adapter)
    adapter.close()
