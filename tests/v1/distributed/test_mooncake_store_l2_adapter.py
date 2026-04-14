# SPDX-License-Identifier: Apache-2.0
"""
Tests for MooncakeStoreL2AdapterConfig and factory registration.

Integration tests require the C++ Mooncake extension and a running
Mooncake Store service.  They are skipped automatically when the
extension is not available.
"""

# Standard
import os
import select

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.mooncake_store_l2_adapter import (
    MooncakeStoreL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

# =============================================================================
# Helpers
# =============================================================================


def _native_mooncake_available() -> bool:
    """Check if the C++ Mooncake extension can be imported."""
    try:
        # First Party
        from lmcache.lmcache_mooncake import LMCacheMooncakeClient  # noqa: F401

        return True
    except ImportError:
        return False


requires_mooncake = pytest.mark.skipif(
    not _native_mooncake_available(),
    reason="C++ Mooncake extension (lmcache_mooncake) not available",
)


def create_object_key(chunk_id: int, model_name: str = "test_model") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
    )


def create_memory_obj(size: int = 256, fill_value: float = 1.0) -> TensorMemoryObj:
    raw_data = torch.empty(size, dtype=torch.float32)
    raw_data.fill_(fill_value)
    metadata = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def wait_for_event_fd(event_fd: int, timeout: float = 10.0) -> bool:
    poll = select.poll()
    poll.register(event_fd, select.POLLIN)
    events = poll.poll(timeout * 1000)
    if events:
        try:
            os.eventfd_read(event_fd)
        except BlockingIOError:
            pass
        return True
    return False


# =============================================================================
# Config Unit Tests (no C++ extension needed)
# =============================================================================


class TestMooncakeStoreL2AdapterConfig:
    """Unit tests for MooncakeStoreL2AdapterConfig."""

    def test_from_dict_minimal(self):
        """Minimal dict with only mooncake keys should work."""
        d = {
            "type": "mooncake_store",
            "local_hostname": "192.168.1.1",
            "metadata_server": "etcd://localhost:2379",
            "global_segment_size": "3221225472",
            "local_buffer_size": "1073741824",
            "protocol": "tcp",
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        # LMCache-only keys should be stripped
        assert "type" not in config.setup_config

        # Mooncake keys should be forwarded as strings
        assert config.setup_config["local_hostname"] == "192.168.1.1"
        assert config.setup_config["metadata_server"] == "etcd://localhost:2379"
        assert config.setup_config["protocol"] == "tcp"

        # Default num_workers
        assert config.num_workers == 4

    def test_from_dict_with_num_workers(self):
        """num_workers should be parsed and excluded from setup_config."""
        d = {
            "type": "mooncake_store",
            "num_workers": 8,
            "local_hostname": "10.0.0.1",
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert config.num_workers == 8
        assert "num_workers" not in config.setup_config
        assert config.setup_config["local_hostname"] == "10.0.0.1"

    def test_from_dict_strips_lmcache_only_keys(self):
        """LMCache-only keys (type, num_workers, eviction) should
        not appear in setup_config."""
        d = {
            "type": "mooncake_store",
            "num_workers": 2,
            "eviction": "lru",
            "local_hostname": "host1",
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert "type" not in config.setup_config
        assert "num_workers" not in config.setup_config
        assert "eviction" not in config.setup_config
        assert config.setup_config["local_hostname"] == "host1"

    def test_from_dict_converts_values_to_str(self):
        """Non-string values should be converted to strings."""
        d = {
            "type": "mooncake_store",
            "global_segment_size": 3221225472,
            "local_buffer_size": 1073741824,
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert config.setup_config["global_segment_size"] == "3221225472"
        assert config.setup_config["local_buffer_size"] == "1073741824"

    def test_from_dict_skips_none_values(self):
        """Keys with None values should be excluded from setup_config."""
        d = {
            "type": "mooncake_store",
            "local_hostname": "host1",
            "optional_key": None,
        }
        config = MooncakeStoreL2AdapterConfig.from_dict(d)

        assert "optional_key" not in config.setup_config
        assert config.setup_config["local_hostname"] == "host1"

    def test_from_dict_invalid_num_workers_zero(self):
        """num_workers=0 should raise ValueError."""
        d = {"type": "mooncake_store", "num_workers": 0}
        with pytest.raises(ValueError, match="num_workers"):
            MooncakeStoreL2AdapterConfig.from_dict(d)

    def test_from_dict_invalid_num_workers_negative(self):
        """Negative num_workers should raise ValueError."""
        d = {"type": "mooncake_store", "num_workers": -1}
        with pytest.raises(ValueError, match="num_workers"):
            MooncakeStoreL2AdapterConfig.from_dict(d)

    def test_from_dict_invalid_num_workers_string(self):
        """Non-integer num_workers should raise ValueError."""
        d = {"type": "mooncake_store", "num_workers": "four"}
        with pytest.raises(ValueError, match="num_workers"):
            MooncakeStoreL2AdapterConfig.from_dict(d)

    def test_constructor_copies_setup_config(self):
        """Constructor should copy the setup_config dict."""
        original = {"key": "value"}
        config = MooncakeStoreL2AdapterConfig(setup_config=original)

        # Mutating the original should not affect the config
        original["key"] = "changed"
        assert config.setup_config["key"] == "value"

    def test_help_returns_string(self):
        """help() should return a non-empty string."""
        h = MooncakeStoreL2AdapterConfig.help()
        assert isinstance(h, str)
        assert len(h) > 0


# =============================================================================
# Factory Registration Tests (no C++ extension needed)
# =============================================================================


class TestMooncakeStoreRegistration:
    """Tests for factory and config type registration."""

    def test_mooncake_store_type_registered(self):
        """'mooncake_store' should be in the registered adapter types."""
        assert "mooncake_store" in get_registered_l2_adapter_types()

    def test_config_type_name(self):
        """get_type_name_for_config should return 'mooncake_store'."""
        config = MooncakeStoreL2AdapterConfig(setup_config={})
        name = get_type_name_for_config(config)
        assert name == "mooncake_store"

    def test_factory_raises_without_extension(self):
        """Factory should raise RuntimeError when C++ extension
        is not available."""
        if _native_mooncake_available():
            pytest.skip("C++ Mooncake extension is available")

        config = MooncakeStoreL2AdapterConfig(
            setup_config={"local_hostname": "localhost"},
            num_workers=2,
        )
        with pytest.raises(RuntimeError, match="Mooncake"):
            create_l2_adapter_from_registry(config)


# =============================================================================
# Integration Tests (require C++ Mooncake extension + running service)
# =============================================================================

# Mooncake service connection params from environment
MOONCAKE_LOCAL_HOSTNAME = os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "")
MOONCAKE_METADATA_SERVER = os.environ.get(
    "MOONCAKE_METADATA_SERVER", "etcd://localhost:2379"
)

requires_mooncake_service = pytest.mark.skipif(
    not _native_mooncake_available() or not MOONCAKE_LOCAL_HOSTNAME,
    reason=("C++ Mooncake extension not available or MOONCAKE_LOCAL_HOSTNAME not set"),
)


@requires_mooncake_service
class TestMooncakeStoreIntegration:
    """Integration tests using real Mooncake Store service.

    These tests require:
    1. The C++ Mooncake extension (lmcache_mooncake) to be built
    2. A running Mooncake Store service
    3. MOONCAKE_LOCAL_HOSTNAME environment variable set

    Set environment variables before running:
        export MOONCAKE_LOCAL_HOSTNAME=<your-ip>
        export MOONCAKE_METADATA_SERVER=etcd://<etcd-host>:2379
    """

    @pytest.fixture(autouse=True)
    def setup_adapter(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": MOONCAKE_LOCAL_HOSTNAME,
                "metadata_server": MOONCAKE_METADATA_SERVER,
                "num_workers": 2,
            }
        )
        self.adapter = create_l2_adapter(config)
        yield
        self.adapter.close()

    def test_event_fds_are_distinct(self):
        """Each operation should have a distinct event fd."""
        fds = {
            self.adapter.get_store_event_fd(),
            self.adapter.get_lookup_and_lock_event_fd(),
            self.adapter.get_load_event_fd(),
        }
        assert len(fds) == 3

    def test_store_and_lookup(self):
        """Store objects, then verify lookup finds them."""
        keys = [create_object_key(i) for i in range(5)]
        objs = [create_memory_obj(size=64, fill_value=float(i)) for i in range(5)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Store
        store_tid = self.adapter.submit_store_task(keys, objs)
        assert wait_for_event_fd(store_fd)
        completed = self.adapter.pop_completed_store_tasks()
        assert completed[store_tid] is True

        # Lookup all — should find everything
        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap is not None
        for i in range(5):
            assert bitmap.test(i) is True, f"Key {i} not found in lookup"

        # Unlock
        self.adapter.submit_unlock(keys)

    def test_lookup_nonexistent_keys(self):
        """Lookup for keys not stored should return all zeros."""
        keys = [create_object_key(i + 10000) for i in range(3)]
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap is not None
        for i in range(3):
            assert bitmap.test(i) is False

    def test_full_store_lookup_load_workflow(self):
        """End-to-end: store -> lookup -> load, verify data integrity."""
        key = create_object_key(42)
        store_obj = create_memory_obj(size=512, fill_value=3.14)
        load_obj = create_memory_obj(size=512, fill_value=0.0)

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()
        load_fd = self.adapter.get_load_event_fd()

        # Store
        store_tid = self.adapter.submit_store_task([key], [store_obj])
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid] is True

        # Lookup
        lookup_tid = self.adapter.submit_lookup_and_lock_task([key])
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        assert bitmap.test(0) is True

        # Load
        load_tid = self.adapter.submit_load_task([key], [load_obj])
        assert wait_for_event_fd(load_fd)
        bitmap = self.adapter.query_load_result(load_tid)
        assert bitmap.test(0) is True

        # Verify data integrity
        assert torch.allclose(load_obj.tensor, store_obj.tensor), (
            "Loaded data does not match stored data"
        )

        # Unlock
        self.adapter.submit_unlock([key])

    def test_batch_store_lookup_load(self):
        """Batch workflow with multiple objects."""
        n = 10
        keys = [create_object_key(i + 100) for i in range(n)]
        store_objs = [
            create_memory_obj(size=128, fill_value=float(i * 7)) for i in range(n)
        ]
        load_objs = [create_memory_obj(size=128, fill_value=0.0) for _ in range(n)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()
        load_fd = self.adapter.get_load_event_fd()

        # Store all
        store_tid = self.adapter.submit_store_task(keys, store_objs)
        assert wait_for_event_fd(store_fd)
        assert self.adapter.pop_completed_store_tasks()[store_tid] is True

        # Lookup all
        lookup_tid = self.adapter.submit_lookup_and_lock_task(keys)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)
        for i in range(n):
            assert bitmap.test(i) is True

        # Load all
        load_tid = self.adapter.submit_load_task(keys, load_objs)
        assert wait_for_event_fd(load_fd)
        bitmap = self.adapter.query_load_result(load_tid)
        for i in range(n):
            assert bitmap.test(i) is True
            assert torch.allclose(load_objs[i].tensor, store_objs[i].tensor), (
                f"Data mismatch for key {i}"
            )

        self.adapter.submit_unlock(keys)

    def test_mixed_lookup_existing_and_missing(self):
        """Lookup a mix of stored and non-stored keys."""
        stored_keys = [create_object_key(i + 200) for i in range(3)]
        stored_objs = [create_memory_obj(fill_value=float(i)) for i in range(3)]

        store_fd = self.adapter.get_store_event_fd()
        lookup_fd = self.adapter.get_lookup_and_lock_event_fd()

        # Store first 3
        self.adapter.submit_store_task(stored_keys, stored_objs)
        assert wait_for_event_fd(store_fd)
        self.adapter.pop_completed_store_tasks()

        # Lookup 5 keys (3 stored + 2 missing)
        all_keys = stored_keys + [
            create_object_key(10100),
            create_object_key(10101),
        ]
        lookup_tid = self.adapter.submit_lookup_and_lock_task(all_keys)
        assert wait_for_event_fd(lookup_fd)
        bitmap = self.adapter.query_lookup_and_lock_result(lookup_tid)

        for i in range(3):
            assert bitmap.test(i) is True, f"Stored key {i} should be found"
        assert bitmap.test(3) is False, "Missing key should not be found"
        assert bitmap.test(4) is False, "Missing key should not be found"

        self.adapter.submit_unlock(stored_keys)

    def test_factory_creates_adapter(self):
        """Verify the factory can create a Mooncake Store L2 adapter."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        config = MooncakeStoreL2AdapterConfig.from_dict(
            {
                "type": "mooncake_store",
                "local_hostname": MOONCAKE_LOCAL_HOSTNAME,
                "metadata_server": MOONCAKE_METADATA_SERVER,
                "num_workers": 2,
            }
        )
        adapter = create_l2_adapter(config)
        try:
            # Should have valid event fds
            assert adapter.get_store_event_fd() >= 0
            assert adapter.get_lookup_and_lock_event_fd() >= 0
            assert adapter.get_load_event_fd() >= 0
        finally:
            adapter.close()
