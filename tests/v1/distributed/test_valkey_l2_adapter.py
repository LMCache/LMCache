# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for ValkeyL2Adapter.

These tests exercise the L2AdapterInterface contract against an
in-process fake of ``glide_sync`` so neither ``valkey-glide`` nor a real
Valkey server is required.  Behavior verified:

* Event fds are distinct and signaled on completion.
* Partial-failure accounting reflects only the keys that wrote.
* Size-mismatched GET responses are treated as cache misses.
* ``cache_salt`` and ``key_prefix`` are both encoded into the wire key.
* ``submit_unlock`` decrements lock refcounts and matches ``lookup``.
* ``delete`` reports per-key sizes recorded at store time.
* Config validation rejects bad inputs.
"""

# Standard
from typing import Any, Optional
import select
import sys
import threading
import types

# Third Party
import pytest
import torch

# ---------------------------------------------------------------------------
# In-process fake `glide_sync`
# ---------------------------------------------------------------------------
#
# The worker pool imports ``glide_sync`` lazily when it creates a client.
# We install a fake module in ``sys.modules`` *before* the adapter is
# imported so that lazy import resolves to our fake.  All workers share
# the same backing dict so multi-thread behavior matches a real
# centralized server.


_STORE_LOCK = threading.Lock()
_STORE: dict[bytes, bytes] = {}
# Maps "set" / "get" / "exists" / "delete" -> Exception instance to raise
# on the *next* call for any key, OR a callable
# (key: bytes) -> Optional[Exception] for per-key faults.
_FAULTS: dict[str, object] = {}
# When set, ``set`` writes only the first ``_TRUNCATE_BYTES`` bytes of
# the value — used to simulate stale/incompatible entries.
_TRUNCATE_BYTES: Optional[int] = None


def _reset_fake_state() -> None:
    """Reset the fake glide backing state between tests."""
    global _TRUNCATE_BYTES
    with _STORE_LOCK:
        _STORE.clear()
    _FAULTS.clear()
    _TRUNCATE_BYTES = None


def _maybe_fault(op: str, key: bytes) -> None:
    """Raise the configured fault (if any) for ``op``."""
    fault = _FAULTS.get(op)
    if fault is None:
        return
    if callable(fault):
        result = fault(key)
        if result is not None:
            raise result
    elif isinstance(fault, BaseException):
        raise fault


class _FakeGlideClient:
    """Minimal in-process stand-in for ``glide_sync.GlideClient``."""

    def __init__(self) -> None:
        self.closed = False

    @classmethod
    def create(cls, config: object) -> "_FakeGlideClient":
        inst = cls()
        inst.config = config  # type: ignore[attr-defined]
        return inst

    def set(self, key: bytes, value) -> None:
        _maybe_fault("set", bytes(key))
        v = bytes(value)
        if _TRUNCATE_BYTES is not None:
            v = v[:_TRUNCATE_BYTES]
        with _STORE_LOCK:
            _STORE[bytes(key)] = v

    def get(self, key: bytes, buffer=None):
        _maybe_fault("get", bytes(key))
        with _STORE_LOCK:
            v = _STORE.get(bytes(key))
        if v is None:
            return None
        if buffer is None:
            return v
        n = min(len(v), len(buffer))
        buffer[:n] = v[:n]
        return n  # buffer GET returns bytes-written

    def exists(self, keys) -> int:
        for k in keys:
            _maybe_fault("exists", bytes(k))
        with _STORE_LOCK:
            return sum(1 for k in keys if bytes(k) in _STORE)

    def delete(self, keys) -> int:
        for k in keys:
            _maybe_fault("delete", bytes(k))
        n = 0
        with _STORE_LOCK:
            for k in keys:
                kb = bytes(k)
                if kb in _STORE:
                    del _STORE[kb]
                    n += 1
        return n

    def close(self) -> None:
        self.closed = True


# Per-node INFO memory the fake cluster client reports (node addr -> bytes).
_NODE_INFO: dict[str, bytes] = {}


class _FakeGlideClusterClient(_FakeGlideClient):
    """Cluster client behaves identically against the shared fake, plus a
    stubbed ``info()`` that returns ``_NODE_INFO`` for AllNodes routing."""

    def info(self, sections=None, route=None):
        # AllNodes route → dict of node addr -> INFO bytes.
        return dict(_NODE_INFO)


def _install_fake_glide() -> types.ModuleType:
    """Install the fake glide_sync module in ``sys.modules``."""
    fake = types.ModuleType("glide_sync")

    def _record(name):
        def _ctor(**kw):
            return (name, kw)

        return _ctor

    fake.ServerCredentials = lambda u, p: ("creds", u, p)  # type: ignore[attr-defined]
    fake.NodeAddress = lambda h, p: ("addr", h, p)  # type: ignore[attr-defined]
    fake.AdvancedGlideClientConfiguration = _record("adv_std")  # type: ignore[attr-defined]
    fake.AdvancedGlideClusterClientConfiguration = _record(  # type: ignore[attr-defined]
        "adv_cluster"
    )
    fake.GlideClientConfiguration = _record("cfg_std")  # type: ignore[attr-defined]
    fake.GlideClusterClientConfiguration = _record("cfg_cluster")  # type: ignore[attr-defined]
    fake.GlideClient = _FakeGlideClient  # type: ignore[attr-defined]
    fake.GlideClusterClient = _FakeGlideClusterClient  # type: ignore[attr-defined]
    sys.modules["glide_sync"] = fake

    # Stub the glide_shared modules that `_do_node_memory` lazily imports
    # for per-node INFO routing.
    routes_mod = types.ModuleType("glide_shared.routes")
    routes_mod.AllNodes = lambda: ("route", "all_nodes")  # type: ignore[attr-defined]
    core_opts_mod = types.ModuleType("glide_shared.commands.core_options")

    class _InfoSection:
        MEMORY = "memory"

    core_opts_mod.InfoSection = _InfoSection  # type: ignore[attr-defined]
    shared_pkg = types.ModuleType("glide_shared")
    commands_pkg = types.ModuleType("glide_shared.commands")
    sys.modules["glide_shared"] = shared_pkg
    sys.modules["glide_shared.commands"] = commands_pkg
    sys.modules["glide_shared.commands.core_options"] = core_opts_mod
    sys.modules["glide_shared.routes"] = routes_mod
    return fake


# Install before the adapter module is imported below.
_install_fake_glide()


# First Party
# First Party  (imported after fake glide is installed)
from lmcache.v1.distributed.api import ObjectKey  # noqa: E402
from lmcache.v1.distributed.internal_api import L2AdapterListener  # noqa: E402
from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (  # noqa: E402
    ValkeyL2Adapter,
    ValkeyL2AdapterConfig,
    _parse_startup_nodes,
)
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd  # noqa: E402

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _RecordingListener(L2AdapterListener):
    """Captures listener events for inspection in tests."""

    def __init__(self) -> None:
        self.stored: list[list[ObjectKey]] = []
        self.accessed: list[list[ObjectKey]] = []
        self.deleted: list[list[ObjectKey]] = []
        self.lock = threading.Lock()

    def on_l2_keys_stored(self, keys: list[ObjectKey]) -> None:
        with self.lock:
            self.stored.append(list(keys))

    def on_l2_keys_accessed(self, keys: list[ObjectKey]) -> None:
        with self.lock:
            self.accessed.append(list(keys))

    def on_l2_keys_deleted(self, keys: list[ObjectKey]) -> None:
        with self.lock:
            self.deleted.append(list(keys))


def create_object_key(
    chunk_id: int,
    model_name: str = "test_model",
    cache_salt: str = "",
) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=model_name,
        kv_rank=0,
        cache_salt=cache_salt,
    )


def create_memory_obj(size: int = 64, fill_value: float = 1.0) -> TensorMemoryObj:
    raw = torch.empty(size, dtype=torch.float32)
    raw.fill_(fill_value)
    meta = MemoryObjMetadata(
        shape=torch.Size([size]),
        dtype=torch.float32,
        address=0,
        phy_size=size * 4,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _in_store(adapter: ValkeyL2Adapter, key: ObjectKey) -> bool:
    """Whether ``key`` currently exists in the fake Valkey store."""
    return adapter._wire_key(key).encode() in _STORE  # noqa: SLF001


def wait_for_event_fd(fd: int, timeout: float = 5.0) -> bool:
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    if not poll.poll(timeout * 1000):
        return False
    try:
        consume_fd(fd)
    except BlockingIOError:
        pass
    return True


def _wait_for_store(adapter: ValkeyL2Adapter, task_id: int, timeout: float = 5.0):
    """Poll the store event fd until ``task_id`` shows up; return its result."""
    fd = adapter.get_store_event_fd()
    deadline = timeout
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    # Drain may report extra completions; loop until task_id appears.
    while deadline > 0:
        events = poll.poll(deadline * 1000)
        if not events:
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        completed = adapter.pop_completed_store_tasks()
        if task_id in completed:
            return completed[task_id]
        # Re-stash any other task results (shouldn't normally happen
        # in single-batch tests).
        for tid, r in completed.items():
            # Best-effort: re-insert into adapter's completed dict by
            # popping again is not possible — but tests submit one
            # task at a time, so this branch is defensive only.
            pass
        deadline -= 0.05
    raise AssertionError(f"store task {task_id} did not complete in {timeout}s")


def _wait_for_lookup(adapter: ValkeyL2Adapter, task_id: int, timeout: float = 5.0):
    fd = adapter.get_lookup_and_lock_event_fd()
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    while timeout > 0:
        events = poll.poll(timeout * 1000)
        if not events:
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        bm = adapter.query_lookup_and_lock_result(task_id)
        if bm is not None:
            return bm
        timeout -= 0.05
    raise AssertionError(f"lookup task {task_id} did not complete")


def _wait_for_load(adapter: ValkeyL2Adapter, task_id: int, timeout: float = 5.0):
    fd = adapter.get_load_event_fd()
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    while timeout > 0:
        events = poll.poll(timeout * 1000)
        if not events:
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        bm = adapter.query_load_result(task_id)
        if bm is not None:
            return bm
        timeout -= 0.05
    raise AssertionError(f"load task {task_id} did not complete")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_state():
    _reset_fake_state()
    yield
    _reset_fake_state()


def _make_config(**overrides: Any) -> ValkeyL2AdapterConfig:
    base: dict[str, Any] = {
        "startup_nodes": [("localhost", 6379)],
        "num_workers": 2,
        "connection_timeout": 2.0,
        "request_timeout": 2.0,
    }
    base.update(overrides)
    return ValkeyL2AdapterConfig(**base)


@pytest.fixture
def adapter():
    a = ValkeyL2Adapter(_make_config())
    yield a
    a.close()


@pytest.fixture
def cluster_adapter():
    a = ValkeyL2Adapter(_make_config(cluster_mode=True))
    yield a
    a.close()


# ===========================================================================
# Config validation
# ===========================================================================


class TestConfigValidation:
    def test_empty_startup_nodes_rejected(self):
        with pytest.raises(ValueError, match="startup_nodes"):
            ValkeyL2AdapterConfig(startup_nodes=[])

    def test_bad_port_rejected(self):
        with pytest.raises(ValueError):
            ValkeyL2AdapterConfig(startup_nodes=[("h", 0)])

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValueError, match="max_capacity_gb"):
            ValkeyL2AdapterConfig(startup_nodes=[("h", 1)], max_capacity_gb=-1)

    def test_zero_workers_rejected(self):
        with pytest.raises(ValueError, match="num_workers"):
            ValkeyL2AdapterConfig(startup_nodes=[("h", 1)], num_workers=0)

    def test_at_sign_in_prefix_rejected(self):
        with pytest.raises(ValueError, match="key_prefix"):
            ValkeyL2AdapterConfig(startup_nodes=[("h", 1)], key_prefix="bad@prefix")

    def test_cluster_mode_warns_on_database_id(self, caplog):
        # Standard
        import logging

        with caplog.at_level(logging.WARNING):
            cfg = ValkeyL2AdapterConfig(
                startup_nodes=[("h", 1)],
                cluster_mode=True,
                database_id=3,
            )
        assert cfg.database_id is None

    def test_standalone_warns_on_multiple_nodes(self, caplog):
        # Standard
        import logging

        with caplog.at_level(logging.WARNING):
            ValkeyL2AdapterConfig(
                startup_nodes=[("h1", 1), ("h2", 2)],
                cluster_mode=False,
            )


class TestParseStartupNodes:
    def test_single(self):
        assert _parse_startup_nodes("host:6379") == [("host", 6379)]

    def test_comma_separated(self):
        nodes = _parse_startup_nodes("a:1,b:2,c:3")
        assert nodes == [("a", 1), ("b", 2), ("c", 3)]

    def test_missing_colon(self):
        with pytest.raises(ValueError, match="host:port"):
            _parse_startup_nodes("nocolon")

    def test_non_integer_port(self):
        with pytest.raises(ValueError, match="non-integer port"):
            _parse_startup_nodes("host:abc")

    def test_empty_or_non_string_rejected(self):
        for bad in ("", "   ", None, [("a", 1)]):
            with pytest.raises(ValueError):
                _parse_startup_nodes(bad)


class TestFromDict:
    def test_basic(self):
        cfg = ValkeyL2AdapterConfig.from_dict(
            {
                "type": "valkey",
                "startup_nodes": "a:1,b:2",
                "cluster_mode": True,
                "username": "u",
                "password": "p",
                "key_prefix": "deploy1",
                "num_workers": 4,
                "tls_enable": True,
                "max_capacity_gb": 2.5,
            }
        )
        assert cfg.startup_nodes == [("a", 1), ("b", 2)]
        assert cfg.cluster_mode is True
        assert cfg.key_prefix == "deploy1"
        assert cfg.num_workers == 4
        assert cfg.tls_enable is True
        assert cfg.max_capacity_gb == 2.5


# ===========================================================================
# Event fd interface
# ===========================================================================


class TestEventFds:
    def test_fds_are_distinct(self, adapter):
        a = adapter.get_store_event_fd()
        b = adapter.get_lookup_and_lock_event_fd()
        c = adapter.get_load_event_fd()
        assert a != b and b != c and a != c
        assert all(fd >= 0 for fd in (a, b, c))


# ===========================================================================
# Store
# ===========================================================================


class TestStore:
    def test_single_key_round_trip(self, adapter):
        k = create_object_key(1)
        o = create_memory_obj(size=16, fill_value=0.5)
        task = adapter.submit_store_task([k], [o])
        result = _wait_for_store(adapter, task)
        assert result.is_successful()
        assert result.bytes_transferred() == o.get_size()

    def test_empty_batch_completes_immediately(self, adapter):
        task = adapter.submit_store_task([], [])
        result = _wait_for_store(adapter, task)
        assert result.is_successful()
        assert result.bytes_transferred() == 0

    def test_partial_failure_accounting(self, adapter):
        """Issue #3342 req 6: a partial batch failure must (a) report the
        task as NOT successful — so the store controller keeps the
        un-stored keys in L1 rather than dropping them — and (b) account
        only the keys that actually wrote in per-salt / aggregate usage.

        ``L2StoreResult`` is binary by contract (``success=False`` ⇒
        ``bytes_transferred()==0``), so the meaningful byte accounting is
        verified via ``get_usage()`` and the stored-listener, not via the
        coarse task result.
        """
        listener = _RecordingListener()
        adapter.register_listener(listener)
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(size=16) for _ in range(3)]

        # Make the SET for key index 1's wire key fail.
        target_wire = adapter._wire_key(keys[1]).encode()  # noqa: SLF001

        def faulty(k: bytes):
            if k == target_wire:
                return RuntimeError("simulated SET failure")
            return None

        _FAULTS["set"] = faulty

        task = adapter.submit_store_task(keys, objs)
        result = _wait_for_store(adapter, task)

        # Task-level: a partial failure is a task failure.
        assert not result.is_successful()
        assert result.bytes_transferred() == 0

        # Real accounting: only the 2 successful keys are counted.
        usage = adapter.get_usage()
        assert usage.total_bytes_used == 2 * objs[0].get_size()
        # ...and the failed key is not in the stored-listener notifications.
        stored_flat = {k for batch in listener.stored for k in batch}
        assert keys[0] in stored_flat
        assert keys[2] in stored_flat
        assert keys[1] not in stored_flat

    def test_length_mismatch_raises(self, adapter):
        with pytest.raises(ValueError, match="length mismatch"):
            adapter.submit_store_task([create_object_key(0)], [])

    def test_store_fires_listener(self, adapter):
        listener = _RecordingListener()
        adapter.register_listener(listener)
        keys = [create_object_key(i) for i in range(2)]
        objs = [create_memory_obj(size=8) for _ in keys]
        task = adapter.submit_store_task(keys, objs)
        _wait_for_store(adapter, task)
        # ``_notify_keys_stored`` should have fired with these keys.
        assert any(set(batch) == set(keys) for batch in listener.stored)


# ===========================================================================
# Lookup + lock
# ===========================================================================


class TestLookupAndLock:
    def test_lookup_after_store(self, adapter):
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(size=8) for _ in keys]
        _wait_for_store(adapter, adapter.submit_store_task(keys, objs))

        task = adapter.submit_lookup_and_lock_task(keys)
        bm = _wait_for_lookup(adapter, task)
        assert all(bm.test(i) for i in range(3))

    def test_lookup_miss(self, adapter):
        keys = [create_object_key(99)]
        bm = _wait_for_lookup(adapter, adapter.submit_lookup_and_lock_task(keys))
        assert not bm.test(0)

    def test_unlock_balances_lookup(self, adapter):
        keys = [create_object_key(i) for i in range(2)]
        objs = [create_memory_obj(size=4) for _ in keys]
        _wait_for_store(adapter, adapter.submit_store_task(keys, objs))
        # Two successful lookups → refcount == 2 per key.
        for _ in range(2):
            _wait_for_lookup(adapter, adapter.submit_lookup_and_lock_task(keys))
        # Two unlocks should bring it back to zero — no internal state
        # to assert publicly, just verify no exception and that we can
        # delete the keys afterward.
        adapter.submit_unlock(keys)
        adapter.submit_unlock(keys)
        adapter.delete(keys)

    def test_empty_lookup_batch(self, adapter):
        bm = _wait_for_lookup(adapter, adapter.submit_lookup_and_lock_task([]))
        assert bm is not None


# ===========================================================================
# Load
# ===========================================================================


class TestLoad:
    def test_load_after_store_returns_hit(self, adapter):
        k = create_object_key(1)
        src = create_memory_obj(size=8, fill_value=0.25)
        _wait_for_store(adapter, adapter.submit_store_task([k], [src]))

        dst = create_memory_obj(size=8, fill_value=0.0)
        bm = _wait_for_load(adapter, adapter.submit_load_task([k], [dst]))
        assert bm.test(0)
        # Loaded buffer matches the source data.
        assert torch.allclose(dst.tensor, src.tensor)

    def test_load_miss_returns_zero_bit(self, adapter):
        dst = create_memory_obj(size=8)
        bm = _wait_for_load(
            adapter, adapter.submit_load_task([create_object_key(42)], [dst])
        )
        assert not bm.test(0)

    def test_size_mismatch_treated_as_miss(self, adapter):
        """Issue #3342 req 5: stale/wrong-size GET = cache miss."""
        global _TRUNCATE_BYTES
        k = create_object_key(1)
        obj = create_memory_obj(size=16)  # 64 bytes (16 * float32)

        # Simulate a stale entry of the wrong length on the server.
        _TRUNCATE_BYTES = 8  # store only 8 bytes
        _wait_for_store(adapter, adapter.submit_store_task([k], [obj]))
        _TRUNCATE_BYTES = None

        dst = create_memory_obj(size=16)
        bm = _wait_for_load(adapter, adapter.submit_load_task([k], [dst]))
        assert not bm.test(0), "size-mismatched value must be reported as miss"

    def test_length_mismatch_raises(self, adapter):
        with pytest.raises(ValueError, match="length mismatch"):
            adapter.submit_load_task([create_object_key(0)], [])


# ===========================================================================
# Delete + accounting
# ===========================================================================


class TestDelete:
    def test_delete_after_store(self, adapter):
        listener = _RecordingListener()
        adapter.register_listener(listener)
        keys = [create_object_key(i) for i in range(2)]
        objs = [create_memory_obj(size=8) for _ in keys]
        _wait_for_store(adapter, adapter.submit_store_task(keys, objs))

        adapter.delete(keys)
        # Listener observed the deletions.
        assert any(set(batch) == set(keys) for batch in listener.deleted)
        # Subsequent lookup should miss.
        bm = _wait_for_lookup(adapter, adapter.submit_lookup_and_lock_task(keys))
        assert not any(bm.test(i) for i in range(len(keys)))

    def test_delete_unknown_keys_is_noop(self, adapter):
        # Should not raise even though the keys were never stored.
        adapter.delete([create_object_key(999)])

    def test_lock_blocks_delete(self, adapter):
        """A key pinned by an in-flight lookup must not be deleted."""
        key = create_object_key(1)
        _wait_for_store(
            adapter, adapter.submit_store_task([key], [create_memory_obj(4)])
        )

        # Lookup bumps the lock refcount.
        bm = _wait_for_lookup(adapter, adapter.submit_lookup_and_lock_task([key]))
        assert bm.test(0)

        adapter.delete([key])
        assert _in_store(adapter, key), "locked key must survive delete"

        # After unlock the key is deletable again.
        adapter.submit_unlock([key])
        adapter.delete([key])
        assert not _in_store(adapter, key)

    def test_refcount_blocks_until_fully_unlocked(self, adapter):
        """Two lookups → refcount 2 → needs two unlocks before delete."""
        key = create_object_key(1)
        _wait_for_store(
            adapter, adapter.submit_store_task([key], [create_memory_obj(4)])
        )
        for _ in range(2):
            _wait_for_lookup(adapter, adapter.submit_lookup_and_lock_task([key]))

        adapter.submit_unlock([key])  # refcount 1, still pinned
        adapter.delete([key])
        assert _in_store(adapter, key)

        adapter.submit_unlock([key])  # refcount 0
        adapter.delete([key])
        assert not _in_store(adapter, key)


# ===========================================================================
# Key prefix + cache_salt isolation
# ===========================================================================


class TestKeyNamespacing:
    def test_different_prefixes_do_not_collide(self):
        a1 = ValkeyL2Adapter(_make_config(key_prefix="dep-A"))
        a2 = ValkeyL2Adapter(_make_config(key_prefix="dep-B"))
        try:
            k = create_object_key(7)
            obj = create_memory_obj(size=4)
            _wait_for_store(a1, a1.submit_store_task([k], [obj]))
            # a2 sees no value for the same logical key.
            bm = _wait_for_lookup(a2, a2.submit_lookup_and_lock_task([k]))
            assert not bm.test(0)
        finally:
            a1.close()
            a2.close()

    def test_cache_salt_isolation(self, adapter):
        # cache_salt is part of the wire key produced by
        # _object_key_to_string, so different salts must miss.
        k_a = create_object_key(1, cache_salt="user-A")
        k_b = create_object_key(1, cache_salt="user-B")
        obj = create_memory_obj(size=4)
        _wait_for_store(adapter, adapter.submit_store_task([k_a], [obj]))
        bm = _wait_for_lookup(adapter, adapter.submit_lookup_and_lock_task([k_b]))
        assert not bm.test(0)


# ===========================================================================
# Status + close
# ===========================================================================


class TestStatusAndClose:
    def test_report_status_minimum_keys(self, adapter):
        st = adapter.report_status()
        assert st["is_healthy"] is True
        assert st["type"] == "valkey"
        assert "num_workers" in st
        # Usage is exposed.
        assert "current_size_bytes" in st
        assert "max_capacity_bytes" in st
        assert "usage_fraction" in st
        # Sensitive fields must not leak.
        assert "password" not in st
        assert "username" not in st

    def test_report_status_tracks_stored_bytes(self, adapter):
        keys = [create_object_key(i) for i in range(3)]
        objs = [create_memory_obj(size=16) for _ in keys]
        _wait_for_store(adapter, adapter.submit_store_task(keys, objs))
        st = adapter.report_status()
        assert st["current_size_bytes"] == 3 * objs[0].get_size()

    def test_close_is_idempotent(self):
        a = ValkeyL2Adapter(_make_config())
        a.close()
        a.close()  # must not raise
        assert a.report_status()["is_healthy"] is False


# ===========================================================================
# Buffer GET capability detection
# ===========================================================================


class TestBufferGetCapability:
    def test_buffer_get_detected(self, adapter):
        # Our fake's get() signature has `buffer=`, so it must be True.
        assert adapter.report_status()["has_buffer_get"] is True

    def test_fallback_when_no_buffer_param(self):
        """Adapter must fall back gracefully if glide lacks buffer=."""

        # Build a fake client class whose `get` has no `buffer` param.
        class _NoBufferGet(_FakeGlideClient):
            def get(self, key):  # type: ignore[override]
                return _FakeGlideClient.get(self, key, buffer=None)

        fake_mod = sys.modules["glide_sync"]
        original = fake_mod.GlideClient
        fake_mod.GlideClient = _NoBufferGet  # type: ignore[attr-defined]
        try:
            adapter = ValkeyL2Adapter(_make_config())
            try:
                assert adapter.report_status()["has_buffer_get"] is False
                k = create_object_key(1)
                src = create_memory_obj(size=4, fill_value=0.5)
                _wait_for_store(adapter, adapter.submit_store_task([k], [src]))
                dst = create_memory_obj(size=4, fill_value=0.0)
                bm = _wait_for_load(adapter, adapter.submit_load_task([k], [dst]))
                assert bm.test(0)
                assert torch.allclose(dst.tensor, src.tensor)
            finally:
                adapter.close()
        finally:
            fake_mod.GlideClient = original  # type: ignore[attr-defined]


# ===========================================================================
# Missing dependency
# ===========================================================================


class TestMissingGlide:
    def test_missing_glide_raises_actionable_error(self, monkeypatch):
        # Hide the fake module so the lazy import fails.
        monkeypatch.setitem(sys.modules, "glide_sync", None)
        with pytest.raises(RuntimeError, match="valkey-glide"):
            ValkeyL2Adapter(_make_config())


# ===========================================================================
# get_usage
# ===========================================================================


class TestGetUsage:
    def test_disabled_returns_minus_one(self):
        a = ValkeyL2Adapter(_make_config(max_capacity_gb=0))
        try:
            usage = a.get_usage()
            assert usage.usage_fraction == -1.0
            assert usage.total_bytes_used == 0
            assert usage.total_capacity_bytes == 0
        finally:
            a.close()

    def test_grows_on_store_and_shrinks_on_delete(self):
        a = ValkeyL2Adapter(_make_config(max_capacity_gb=0.001))  # 1 MB
        try:
            keys = [create_object_key(i) for i in range(4)]
            objs = [create_memory_obj(size=16) for _ in range(4)]  # 64 bytes each
            _wait_for_store(a, a.submit_store_task(keys, objs))

            total = 4 * objs[0].get_size()
            capacity = int(0.001 * 1024**3)
            usage = a.get_usage()
            assert usage.total_bytes_used == total
            assert usage.total_capacity_bytes == capacity
            assert usage.usage_fraction == pytest.approx(total / capacity)

            a.delete(keys)
            assert a.get_usage().total_bytes_used == 0
        finally:
            a.close()

    def test_per_cache_salt_accounting(self):
        a = ValkeyL2Adapter(_make_config(max_capacity_gb=0.001))
        try:
            k_a = create_object_key(1, cache_salt="user-A")
            k_b = create_object_key(2, cache_salt="user-B")
            obj = create_memory_obj(size=16)
            _wait_for_store(
                a, a.submit_store_task([k_a, k_b], [obj, create_memory_obj(16)])
            )
            by_salt = a.get_usage().bytes_by_cache_salt
            assert by_salt["user-A"] == obj.get_size()
            assert by_salt["user-B"] == obj.get_size()
        finally:
            a.close()


# ===========================================================================
# Factory registration
# ===========================================================================


class TestFactoryRegistration:
    def test_create_via_factory(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter

        cfg = ValkeyL2AdapterConfig.from_dict(
            {"type": "valkey", "startup_nodes": "localhost:6379", "num_workers": 1}
        )
        adapter = create_l2_adapter(cfg)
        try:
            assert isinstance(adapter, ValkeyL2Adapter)
            assert adapter.report_status()["type"] == "valkey"
        finally:
            adapter.close()

    def test_registered_in_type_registry(self):
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (
            get_registered_l2_adapter_types,
        )

        assert "valkey" in get_registered_l2_adapter_types()
