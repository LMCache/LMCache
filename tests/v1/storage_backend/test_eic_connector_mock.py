# SPDX-License-Identifier: Apache-2.0
"""
Device-free tests for the EIC remote connector.

The vendor ``eic`` client is not on public PyPI and libcudart is absent from
CPU-only / Ascend images, so both are faked here. These tests cover behaviour
that needs no live EIC cluster and no CUDA toolkit; connector-against-cluster
behaviour stays in test_eic.py, which requires a real ``eic`` install.
"""

# Standard
from types import SimpleNamespace
import asyncio
import ctypes
import importlib
import sys
import types
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# lmcache.v1.memory_management imports the compiled extensions at module
# scope. Without a built install they are never used on these code paths, so
# stub them before any lmcache storage module gets imported.
try:
    # First Party
    import lmcache.c_ops  # noqa: F401
except ImportError:
    sys.modules.setdefault("lmcache.c_ops", MagicMock(name="lmcache.c_ops"))

# First Party
import lmcache  # noqa: E402

if not hasattr(lmcache, "device_ops"):
    lmcache.device_ops = MagicMock(name="device_ops")
try:
    # First Party
    import lmcache.lmcache_native  # noqa: F401
except ImportError:
    sys.modules.setdefault(
        "lmcache.lmcache_native", MagicMock(name="lmcache.lmcache_native")
    )

from lmcache.utils import get_size_bytes  # noqa: E402
from lmcache.v1.memory_management import (  # noqa: E402
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.protocol import (  # noqa: E402
    DTYPE_TO_INT,
    INT_TO_DTYPE,
    RemoteMetadata,
    get_remote_metadata_bytes,
    init_remote_metadata_info,
)

init_remote_metadata_info(1)

# The shared 5 GB pinned-memory allocator in tests/conftest.py needs a built
# native extension; these tests allocate plain CPU tensors themselves.
pytestmark = pytest.mark.no_shared_allocator


class _StatusCode:
    SUCCESS = 0
    KEY_NOT_EXIST = 1
    PARTIAL_FAILED = 2
    FAILED = 3


class _StringVector:
    def __init__(self):
        self._items = []

    def append(self, item):
        self._items.append(item)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        return self._items[index]


class _IOBuffers:
    def __init__(self):
        self.entries = []

    def append(self, ptr, size, need_cuda_copy=False):
        self.entries.append((ptr, size, need_cuda_copy))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.entries[index]


def _overall_status(codes):
    if all(code == _StatusCode.SUCCESS for code in codes):
        return _StatusCode.SUCCESS
    if all(code != _StatusCode.SUCCESS for code in codes):
        return _StatusCode.FAILED
    return _StatusCode.PARTIAL_FAILED


class _FakeClient:
    def __init__(self):
        # base key -> RemoteMetadata / payload bytes; absence means a miss.
        self.meta_script = {}
        self.data_script = {}
        # (overall_status, per_key_codes) or a callable(keys, vals, option).
        self.mset_result = None
        self.mset_handler = None
        # When set, overrides mget entirely: fn(keys, vals) -> (status, codes).
        self.mget_override = None
        self.mget_calls = []
        self.mset_calls = []

    def init(self, instance_id, endpoint, option):
        self.instance_id = instance_id
        self.endpoint = endpoint
        return 0

    def mget(self, keys, option, vals):
        keys = list(keys)
        if self.mget_override is not None:
            status, codes = self.mget_override(keys, vals)
            self.mget_calls.append(("override", keys, status))
            return status, vals, SimpleNamespace(status_codes=list(codes))

        stage = "meta" if keys and keys[0].endswith("_meta") else "data"
        codes = []
        for idx, key in enumerate(keys):
            base = key[: -len("_meta")] if stage == "meta" else key
            if stage == "meta":
                payload_obj = self.meta_script.get(base)
            else:
                payload_obj = self.data_script.get(base)
            if payload_obj is None:
                codes.append(_StatusCode.KEY_NOT_EXIST)
                continue
            payload = payload_obj.serialize() if stage == "meta" else payload_obj
            ptr, size, _ = vals.entries[idx]
            ctypes.memmove(ptr, payload, min(size, len(payload)))
            codes.append(_StatusCode.SUCCESS)

        status = _overall_status(codes)
        self.mget_calls.append((stage, keys, status, vals.entries))
        return status, vals, SimpleNamespace(status_codes=codes)

    def mset(self, keys, vals, option):
        keys = list(keys)
        self.mset_calls.append((keys, vals.entries, option))
        if self.mset_handler is not None:
            return self.mset_handler(keys, vals, option)
        if self.mset_result is not None:
            status, codes = self.mset_result
            return status, SimpleNamespace(status_codes=list(codes))
        return _StatusCode.SUCCESS, SimpleNamespace(
            status_codes=[_StatusCode.SUCCESS] * len(keys)
        )

    def mexist(self, keys, option):
        # Prebuilt connection just probes reachability; answer miss.
        return _StatusCode.SUCCESS, SimpleNamespace(
            status_codes=[_StatusCode.KEY_NOT_EXIST]
        )

    def register_memory(self, vals, meminfo):
        return True


def build_fake_eic():
    """Build a fresh fake ``eic`` module with fresh client state."""
    import enum

    class TransportType(enum.IntEnum):
        TRANSPORT_RDMA = 2
        TRANSPORT_GDR = 3

    class LogLevel(enum.IntEnum):
        INFO = 1

    class MemoryType(enum.IntEnum):
        MEMORY_CUDA = 1

    client_cls = _FakeClient

    fake = types.ModuleType("eic")
    fake.Client = client_cls
    fake.InitOption = type("InitOption", (), {})
    fake.SetOption = type("SetOption", (), {})
    fake.GetOption = type("GetOption", (), {})
    fake.ExistOption = type("ExistOption", (), {})
    fake.StringVector = _StringVector
    fake.IOBuffers = _IOBuffers
    fake.MemoryInfo = type("MemoryInfo", (), {})
    fake.StatusCode = _StatusCode
    fake.TransportType = TransportType
    fake.LogLevel = LogLevel
    fake.MemoryType = MemoryType
    return fake


class _FakeLoop:
    # AsyncPQExecutor only schedules onto the loop at construction; the worker
    # coroutines never need to run in these tests.
    def call_soon_threadsafe(self, callback, *args):
        pass


class FakeKey:
    def __init__(self, key_str):
        self.key_str = key_str

    def to_string(self):
        return self.key_str


def make_memory_obj(shapes, dtypes, fmt=MemoryFormat.KV_2LTD, fill=1):
    shapes = [torch.Size(s) for s in shapes]
    nbytes = get_size_bytes(shapes, dtypes)
    raw = torch.full((nbytes,), fill & 0xFF, dtype=torch.uint8)
    metadata = MemoryObjMetadata(
        shape=shapes[0],
        dtype=dtypes[0],
        address=raw.data_ptr(),
        phy_size=nbytes,
        ref_count=1,
        pin_count=0,
        fmt=fmt,
        shapes=shapes,
        dtypes=list(dtypes),
    )
    return TensorMemoryObj(raw, metadata, None)


class AddrlessMemoryObj:
    """MemoryObj stand-in with no addressable buffer."""

    def __init__(self, num_groups=1):
        self._shapes = [torch.Size([1, 1, 1, 1])] * num_groups
        self._dtypes = [torch.uint8] * num_groups

    @property
    def byte_array(self):
        return b""

    def get_shapes(self):
        return self._shapes

    def get_dtypes(self):
        return self._dtypes

    def get_memory_format(self):
        return MemoryFormat.KV_2LTD

    def get_physical_size(self):
        return 0

    @property
    def tensor(self):
        return None

    def get_tensor(self, index):
        return None


class FakeAllocator:
    def __init__(self):
        self.config = None
        self.metadata = None
        self.allocations = []

    def allocate(self, shapes, dtypes, fmt):
        obj = make_memory_obj(list(shapes), list(dtypes), fmt)
        self.allocations.append(obj)
        return obj


@pytest.fixture
def fake_eic(monkeypatch):
    fake = build_fake_eic()
    monkeypatch.setitem(sys.modules, "eic", fake)
    mod = importlib.import_module("lmcache.v1.storage_backend.connector.eic_connector")
    importlib.reload(mod)
    # Skip the metadata-driven RemoteConnector.__init__; the tests set the
    # attributes the connector methods read directly.
    monkeypatch.setattr(
        mod.RemoteConnector, "__init__", lambda self, config, metadata: None
    )

    class _FakePQExecutor:
        def __init__(self, loop):
            pass

        async def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(mod, "AsyncPQExecutor", _FakePQExecutor)
    return SimpleNamespace(module=mod, client_cls=fake.Client)


def write_config(tmp_path, monkeypatch, trans_type=2):
    config_path = tmp_path / "lmcache_eic.yaml"
    config_path.write_text(
        "remote_url: 'eic://127.0.0.1:12500'\n"
        "eic_instance_id: 'test-instance'\n"
        f"eic_trans_type: {trans_type}\n"
        "eic_thread_num: 1\n"
        f"eic_log_dir: {tmp_path / 'eic_log'}\n"
        "eic_log_level: 1\n"
        "eic_kv_ttl: -1\n"
        "eic_kv_ns: ''\n"
    )
    monkeypatch.setenv("LMCACHE_CONFIG_FILE", str(config_path))


@pytest.fixture
def env(fake_eic, monkeypatch, tmp_path):
    write_config(tmp_path, monkeypatch, trans_type=2)
    conn = fake_eic.module.EICConnector(
        "eic://127.0.0.1:12500/", _FakeLoop(), FakeAllocator()
    )
    conn.remote_metadata_bytes = get_remote_metadata_bytes()
    return SimpleNamespace(conn=conn, client=conn.connection, mod=fake_eic.module)


def test_missing_libcudart_rdma_still_constructs(fake_eic, monkeypatch, tmp_path):
    real_cdll = ctypes.CDLL

    def raising_cdll(name, *args, **kwargs):
        if name == "libcudart.so":
            raise OSError("libcudart.so: cannot open shared object file")
        return real_cdll(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "CDLL", raising_cdll)
    write_config(tmp_path, monkeypatch, trans_type=2)

    conn = fake_eic.module.EICConnector(
        "eic://127.0.0.1:12500/", _FakeLoop(), FakeAllocator()
    )
    # RDMA works without a CUDA runtime; nothing to bind cudaMemcpy onto.
    assert conn.cuda_lib is None


def test_missing_libcudart_rejects_gdr(fake_eic, monkeypatch, tmp_path):
    real_cdll = ctypes.CDLL

    def raising_cdll(name, *args, **kwargs):
        if name == "libcudart.so":
            raise OSError("libcudart.so: cannot open shared object file")
        return real_cdll(name, *args, **kwargs)

    monkeypatch.setattr(ctypes, "CDLL", raising_cdll)
    write_config(tmp_path, monkeypatch, trans_type=3)

    with pytest.raises(RuntimeError, match="libcudart"):
        fake_eic.module.EICConnector(
            "eic://127.0.0.1:12500/", _FakeLoop(), FakeAllocator()
        )


def _meta_for(shape, dtype=torch.bfloat16):
    shape = torch.Size(shape)
    return RemoteMetadata(
        get_size_bytes([shape], [dtype]),
        [shape],
        [dtype],
        MemoryFormat.KV_2LTD,
    )


SHAPE = (1, 2, 2, 2)  # 16 bfloat16 bytes


def test_batched_get_all_hit_two_mgets(env):
    client = env.client
    connector = env.conn
    keys = [FakeKey("k0"), FakeKey("k1")]
    for key, fill in (("k0", 0xAB), ("k1", 0xCD)):
        client.meta_script[key] = _meta_for(SHAPE)
        client.data_script[key] = bytes([fill]) * 16

    results = asyncio.run(connector._batched_get(keys))

    assert len(results) == 2
    assert results[0] is not None and results[1] is not None
    assert bytes(results[0].raw_data[:16].tolist()) == bytes([0xAB]) * 16
    assert bytes(results[1].raw_data[:16].tolist()) == bytes([0xCD]) * 16
    # One meta mget and one data mget for the whole batch, not 2N serial ones.
    assert len(client.mget_calls) == 2
    assert client.mget_calls[0][0] == "meta"
    assert len(client.mget_calls[0][1]) == 2
    assert client.mget_calls[1][0] == "data"
    assert len(client.mget_calls[1][1]) == 2
    # RDMA never asks for a CUDA copy.
    assert all(entry[2] is False for entry in client.mget_calls[1][3])


def test_batched_get_meta_miss_keeps_slot(env):
    client = env.client
    connector = env.conn
    keys = [FakeKey("k0"), FakeKey("k1"), FakeKey("k2")]
    client.meta_script["k0"] = _meta_for(SHAPE)
    client.data_script["k0"] = b"\x01" * 16
    client.meta_script["k2"] = _meta_for(SHAPE)
    client.data_script["k2"] = b"\x02" * 16

    results = asyncio.run(connector._batched_get(keys))

    assert results[0] is not None
    assert results[1] is None  # miss holds its slot
    assert results[2] is not None

    filtered = asyncio.run(connector._batched_get_non_blocking("lookup", keys))
    # Non-blocking returns the contiguous prefix only: k1 misses, so k2 must
    # not compact forward into the position callers zip against keys.
    assert len(filtered) == 1
    assert bytes(filtered[0].raw_data[:16].tolist()) == b"\x01" * 16


def test_non_blocking_data_miss_hole_truncates_prefix(env):
    # contains() can report hits that fail at data time (eviction between the
    # two calls). A middle data miss must truncate the prefix, not shift later
    # hits forward.
    client = env.client
    connector = env.conn
    keys = [FakeKey("k0"), FakeKey("k1"), FakeKey("k2")]
    for key in ("k0", "k1", "k2"):
        client.meta_script[key] = _meta_for(SHAPE)
    client.data_script["k0"] = b"\x01" * 16
    client.data_script["k2"] = b"\x02" * 16  # k1 data misses

    out = asyncio.run(connector._batched_get_non_blocking("lookup", keys))

    assert len(out) == 1
    assert bytes(out[0].raw_data[:16].tolist()) == b"\x01" * 16
    # Allocations preserve key order: k0 kept, k1 (data miss) released,
    # k2 fetched but dropped past the hole and released.
    alloc = connector.memory_allocator.allocations
    assert alloc[0].get_ref_count() == 1
    assert alloc[1].get_ref_count() == 0
    assert alloc[2].get_ref_count() == 0


def test_batched_get_data_miss_releases_object(env):
    client = env.client
    connector = env.conn
    keys = [FakeKey("k0"), FakeKey("k1")]
    for key in ("k0", "k1"):
        client.meta_script[key] = _meta_for(SHAPE)
    client.data_script["k0"] = b"\x01" * 16
    # k1 meta hits but data misses.

    results = asyncio.run(connector._batched_get(keys))

    assert results[0] is not None
    assert results[1] is None
    assert results[0].get_ref_count() == 1
    # The allocated-but-unfilled object is released, not leaked.
    assert connector.memory_allocator.allocations[1].get_ref_count() == 0


def test_batched_get_all_miss_is_one_mget(env):
    client = env.client
    connector = env.conn
    keys = [FakeKey("k0"), FakeKey("k1")]

    results = asyncio.run(connector._batched_get(keys))

    assert results == [None, None]
    # All-miss at meta stage: no data mget is issued.
    assert len(client.mget_calls) == 1
    assert client.mget_calls[0][0] == "meta"
    assert client.mget_calls[0][2] == _StatusCode.FAILED


def test_batched_get_unknown_meta_status_returns_none(env):
    client = env.client
    connector = env.conn

    def override(keys, vals):
        return 99, [99] * len(keys)

    client.mget_override = override
    results = asyncio.run(connector._batched_get([FakeKey("k0")]))
    assert results == [None]


def test_put_sync_success_and_failure_modes(env):
    client = env.client
    connector = env.conn
    obj = make_memory_obj([SHAPE], [torch.bfloat16])
    key = FakeKey("pk0")

    client.mset_result = (
        _StatusCode.SUCCESS,
        [_StatusCode.SUCCESS, _StatusCode.SUCCESS],
    )
    connector._put_sync(key, obj)
    assert len(client.mset_calls) == 1
    sent_keys, entries, _ = client.mset_calls[0]
    assert sent_keys == ["pk0_meta", "pk0"]
    assert len(entries) == 2

    # A whole-call failure must surface instead of looking like success.
    client.mset_result = (
        _StatusCode.FAILED,
        [_StatusCode.FAILED, _StatusCode.FAILED],
    )
    with pytest.raises(RuntimeError, match="eic mset failed"):
        connector._put_sync(key, obj)

    # PARTIAL_FAILED overall but both of this key's codes succeeded: per-key
    # status_codes are authoritative, so the put counts as success.
    client.mset_result = (
        _StatusCode.PARTIAL_FAILED,
        [_StatusCode.SUCCESS, _StatusCode.SUCCESS],
    )
    connector._put_sync(key, obj)

    # One failed per-key code is a real failure for this key.
    client.mset_result = (
        _StatusCode.PARTIAL_FAILED,
        [_StatusCode.SUCCESS, _StatusCode.FAILED],
    )
    with pytest.raises(RuntimeError):
        connector._put_sync(key, obj)


def test_put_sync_no_address_raises(env):
    connector = env.conn
    with pytest.raises(RuntimeError, match="no address"):
        connector._put_sync(FakeKey("pk0"), AddrlessMemoryObj())


def test_support_batched_put_enabled(env):
    assert env.conn.support_batched_put() is True


def test_batched_put_skips_zero_ptr_keeps_batch(env):
    client = env.client
    connector = env.conn
    keys = [FakeKey("b0"), FakeKey("b1"), FakeKey("b2")]
    objs = [
        make_memory_obj([SHAPE], [torch.bfloat16]),
        AddrlessMemoryObj(),
        make_memory_obj([SHAPE], [torch.bfloat16]),
    ]
    client.mset_result = (
        _StatusCode.SUCCESS,
        [_StatusCode.SUCCESS] * 4,
    )

    asyncio.run(connector._batched_put(keys, objs))

    assert len(client.mset_calls) == 1
    sent_keys, entries, _ = client.mset_calls[0]
    # The addressless middle chunk is skipped; the other two still ship their
    # meta+data pairs.
    assert sent_keys == ["b0_meta", "b0", "b2_meta", "b2"]
    assert len(entries) == 4


def test_batched_put_partial_failed_runs_per_key_loop(env, caplog):
    client = env.client
    connector = env.conn
    keys = [FakeKey("b0"), FakeKey("b1")]
    objs = [
        make_memory_obj([SHAPE], [torch.bfloat16]),
        make_memory_obj([SHAPE], [torch.bfloat16]),
    ]
    client.mset_result = (
        _StatusCode.PARTIAL_FAILED,
        [
            _StatusCode.SUCCESS,
            _StatusCode.SUCCESS,
            _StatusCode.SUCCESS,
            _StatusCode.FAILED,
        ],
    )

    # PARTIAL_FAILED must not early-return: the per-key confirmation loop runs
    # and logs the one failed data key.
    asyncio.run(connector._batched_put(keys, objs))
    assert len(client.mset_calls) == 1
    assert any("b1" in record.getMessage() for record in caplog.records)


def test_batched_put_whole_call_failed_returns(env):
    client = env.client
    connector = env.conn
    keys = [FakeKey("b0"), FakeKey("b1"), FakeKey("b2")]
    objs = [make_memory_obj([SHAPE], [torch.bfloat16]) for _ in range(3)]
    client.mset_result = (
        _StatusCode.FAILED,
        [_StatusCode.FAILED] * 6,
    )

    # Whole-call failure is logged and returned, not raised.
    asyncio.run(connector._batched_put(keys, objs))
    assert len(client.mset_calls) == 1
    assert len(client.mset_calls[0][0]) == 6


def test_transfer_base_ptr_multi_group(fake_eic):
    mod = fake_eic.module
    shapes = [torch.Size([1, 2, 2, 2]), torch.Size([1, 2, 2, 2])]
    dtypes = [torch.bfloat16, torch.bfloat16]
    obj = make_memory_obj(shapes, dtypes)

    # The singular view reshapes the whole buffer with group 0's geometry and
    # raises on a multi-group object; the helper still finds the base address.
    with pytest.raises(RuntimeError):
        _ = obj.tensor
    assert mod._transfer_base_ptr(obj) == obj.raw_data.data_ptr()

    single = make_memory_obj([SHAPE], [torch.bfloat16])
    assert mod._transfer_base_ptr(single) == single.tensor.data_ptr()
    assert mod._transfer_base_ptr(AddrlessMemoryObj(num_groups=2)) == 0


def test_int8_dtype_mapping_id_9():
    assert DTYPE_TO_INT[torch.int8] == 9
    assert INT_TO_DTYPE[9] is torch.int8
    # Existing codes must not move.
    assert DTYPE_TO_INT[torch.bfloat16] == 3
    assert INT_TO_DTYPE[8] is torch.float8_e5m2

    # Round-trips through the remote metadata wire format.
    init_remote_metadata_info(1)
    meta = RemoteMetadata(
        8, [torch.Size([2, 1, 1, 1])], [torch.int8], MemoryFormat.KV_2LTD
    )
    restored = RemoteMetadata.deserialize(meta.serialize())
    assert restored.dtypes == [torch.int8]
    assert restored.shapes == [torch.Size([2, 1, 1, 1])]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
