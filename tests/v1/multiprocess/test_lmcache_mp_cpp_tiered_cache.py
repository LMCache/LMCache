# SPDX-License-Identifier: Apache-2.0
"""Tests for the first LMCache MP C++ mirror slice."""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import socket
import sys
import time

# Third Party
import torch
import zmq

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import CudaIPCWrapper, IPCCacheEngineKey
from lmcache.v1.multiprocess.mq import MessageQueueClient, msgspec_encode
from lmcache.v1.multiprocess.protocol import (
    LMCACHE_MP_PROTOCOL_VERSION,
    RequestType,
    get_protocol_schema,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CPP_PYTHON = _REPO_ROOT / "LMCache-mp-cpp" / "python"
sys.path.insert(0, str(_CPP_PYTHON))

# Third Party
from lmcache_mp_cpp import TieredCache  # noqa: E402
from lmcache_mp_cpp.ipc_key import decode_ipc_key, object_key_strings  # noqa: E402
from lmcache_mp_cpp.key_compat import (  # noqa: E402
    blake3_chunk_hashes,
    blake3_hash_tokens,
    blake3_none_hash,
    compute_kv_rank,
    expand_kv_ranks,
    object_key_string,
)
from lmcache_mp_cpp.l2_adapter import (  # noqa: E402
    FileSystemL2Adapter,
    fs_l2_filename,
)
from lmcache_mp_cpp.protocol_compat import (  # noqa: E402
    protocol_version,
    request_type_name,
    request_type_value,
)
from lmcache_mp_cpp.server import run_cpp_cache_server  # noqa: E402
from lmcache_mp_cpp.storage_manager import (  # noqa: E402
    CxxTieredStorageManager,
    object_key_to_string,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _object_key(index: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(index),
        model_name="facebook/opt-125m",
        kv_rank=0,
    )


def test_cuda_ipc_wrapper_native_friendly_serializer_round_trips_metadata():
    wrapper = CudaIPCWrapper.__new__(CudaIPCWrapper)
    wrapper.handle = (0, b"cuda-handle", 4096, 0, b"ref", 0, b"event", False)
    wrapper.dtype = torch.float16
    wrapper.shape = (2, 4, 16, 8)
    wrapper.stride = (512, 128, 8, 1)
    wrapper.storage_offset = 0
    wrapper.device_uuid = "GPU-fake-uuid"

    decoded = CudaIPCWrapper.Deserialize(CudaIPCWrapper.Serialize(wrapper))

    assert decoded == wrapper


def test_vllm_adapter_wrap_kv_caches_can_use_raw_cuda_ipc(monkeypatch):
    # First Party
    from lmcache.integration.vllm import vllm_multi_process_adapter as mp_adapter

    class FakeCudaIPCWrapper:
        def __init__(self, tensor):
            self.tensor = tensor

    class FakeRawCudaIPCWrapper:
        def __init__(self, tensor):
            self.tensor = tensor

    monkeypatch.setattr(mp_adapter, "CudaIPCWrapper", FakeCudaIPCWrapper)
    monkeypatch.setattr(mp_adapter, "RawCudaIPCWrapper", FakeRawCudaIPCWrapper)
    tensor = torch.empty(1)

    default_wrapped = mp_adapter.wrap_kv_caches({"layer": tensor})
    raw_wrapped = mp_adapter.wrap_kv_caches(
        {"layer": tensor},
        use_raw_cuda_ipc=True,
    )

    assert isinstance(default_wrapped[0], FakeCudaIPCWrapper)
    assert isinstance(raw_wrapped[0], FakeRawCudaIPCWrapper)


def test_tiered_cache_spills_to_disk_and_reads_back(tmp_path):
    with TieredCache(dram_capacity_bytes=8, disk_path=tmp_path) as cache:
        cache.put("k1", b"aaaa")
        cache.put("k2", b"bbbb")
        cache.put("k3", b"cccc")

        stats = cache.stats()
        assert stats.total_entries == 3
        assert stats.dram_bytes == 8
        assert stats.disk_bytes == 4
        assert stats.dram_entries == 2
        assert stats.disk_entries == 1
        assert stats.eviction_count == 1

        assert cache.exists("k1")
        assert cache.get("k1") == b"aaaa"
        stats_after_promote = cache.stats()
        assert stats_after_promote.dram_bytes == 8
        assert stats_after_promote.disk_bytes == 4
        assert stats_after_promote.dram_entries == 2
        assert stats_after_promote.disk_entries == 1
        assert stats_after_promote.eviction_count == 2
        assert cache.get("k2") == b"bbbb"
        assert cache.get("k3") == b"cccc"


def test_tiered_cache_resident_get_refreshes_lru_before_spill(tmp_path):
    with TieredCache(dram_capacity_bytes=8, disk_path=tmp_path) as cache:
        cache.put("old", b"aaaa")
        cache.put("newer", b"bbbb")

        assert cache.get("old") == b"aaaa"
        cache.put("incoming", b"cccc")

        assert cache.is_resident("old")
        assert not cache.is_resident("newer")
        assert cache.is_resident("incoming")
        stats = cache.stats()
        assert stats.dram_bytes == 8
        assert stats.disk_bytes == 4
        assert stats.eviction_count == 1


def test_tiered_cache_disk_promotion_refreshes_lru_before_spill(tmp_path):
    with TieredCache(dram_capacity_bytes=8, disk_path=tmp_path) as cache:
        cache.put("first", b"aaaa")
        cache.put("second", b"bbbb")
        cache.put("third", b"cccc")
        assert not cache.is_resident("first")
        assert cache.is_resident("second")
        assert cache.is_resident("third")

        assert cache.get("first") == b"aaaa"

        assert cache.is_resident("first")
        assert not cache.is_resident("second")
        assert cache.is_resident("third")
        stats = cache.stats()
        assert stats.dram_bytes == 8
        assert stats.disk_bytes == 4
        assert stats.eviction_count == 2


def test_tiered_cache_tracks_exact_unaligned_byte_accounting(tmp_path):
    with TieredCache(dram_capacity_bytes=10, disk_path=tmp_path) as cache:
        cache.put("small", b"abc")
        cache.put("middle", b"defg")
        cache.put("large", b"hijkl")

        stats = cache.stats()
        assert stats.dram_bytes == 9
        assert stats.disk_bytes == 3
        assert stats.dram_entries == 2
        assert stats.disk_entries == 1
        assert stats.total_entries == 3

        assert cache.get("small") == b"abc"
        stats_after_promote = cache.stats()
        assert stats_after_promote.dram_bytes == 8
        assert stats_after_promote.disk_bytes == 4
        assert stats_after_promote.dram_entries == 2
        assert stats_after_promote.disk_entries == 1
        assert stats_after_promote.total_entries == 3

        cache.put("large", b"xy")
        stats_after_replace = cache.stats()
        assert stats_after_replace.dram_bytes == 5
        assert stats_after_replace.disk_bytes == 4
        assert stats_after_replace.dram_entries == 2
        assert stats_after_replace.disk_entries == 1
        assert stats_after_replace.total_entries == 3
        assert cache.size("large") == 2
        assert cache.get("large") == b"xy"


def test_tiered_cache_failed_spill_store_rolls_back_new_entry(tmp_path):
    disk_path = tmp_path / "disk"
    with TieredCache(dram_capacity_bytes=4, disk_path=disk_path) as cache:
        cache.put("old", b"aaaa")
        disk_path.rmdir()

        try:
            cache.put("new", b"bbbb")
        except RuntimeError as exc:
            assert "cannot open" in str(exc)
        else:
            raise AssertionError("store should fail when spill directory is missing")

        assert cache.get("old") == b"aaaa"
        assert not cache.exists("new")
        stats = cache.stats()
        assert stats.dram_bytes == 4
        assert stats.disk_bytes == 0
        assert stats.dram_entries == 1
        assert stats.total_entries == 1
        assert stats.eviction_count == 0


def test_tiered_cache_failed_spill_overwrite_restores_spilled_entry(tmp_path):
    disk_path = tmp_path / "disk"
    with TieredCache(dram_capacity_bytes=4, disk_path=disk_path) as cache:
        cache.put("replace", b"aaaa")
        cache.put("resident", b"bbbb")
        assert not cache.is_resident("replace")
        assert cache.is_resident("resident")
        disk_path.chmod(0o500)

        try:
            cache.put("replace", b"zzzz")
        except RuntimeError as exc:
            assert "cannot open" in str(exc)
        else:
            raise AssertionError("overwrite should fail when spill cannot write")
        finally:
            disk_path.chmod(0o700)

        assert cache.get("replace") == b"aaaa"
        assert cache.get("resident") == b"bbbb"
        stats = cache.stats()
        assert stats.dram_bytes == 4
        assert stats.disk_bytes == 4
        assert stats.dram_entries == 1
        assert stats.disk_entries == 1
        assert stats.total_entries == 2


def test_tiered_cache_duplicate_store_replaces_spilled_entry(tmp_path):
    with TieredCache(dram_capacity_bytes=8, disk_path=tmp_path) as cache:
        cache.put("replace", b"aaaa")
        cache.put("resident-a", b"bbbb")
        cache.put("resident-b", b"cccc")
        assert not cache.is_resident("replace")

        cache.put("replace", b"zz")

        assert cache.get("replace") == b"zz"
        assert cache.size("replace") == 2
        stats = cache.stats()
        assert stats.total_entries == 3
        assert stats.dram_bytes == 6
        assert stats.disk_bytes == 4
        assert stats.dram_entries == 2
        assert stats.disk_entries == 1


def test_tiered_cache_protects_locked_entries_from_spill_and_remove(tmp_path):
    with TieredCache(dram_capacity_bytes=8, disk_path=tmp_path) as cache:
        cache.put("locked", b"aaaa")
        cache.put("victim", b"bbbb")
        assert cache.lock("locked")

        cache.put("new", b"cccc")
        stats = cache.stats()

        assert stats.locked_entries == 1
        assert stats.lock_count == 1
        assert stats.locked_bytes == 4
        assert stats.dram_bytes <= 8
        assert stats.eviction_count == 1
        assert cache.is_resident("locked")
        assert not cache.is_resident("victim")

        try:
            cache.remove("locked")
        except RuntimeError as exc:
            assert "locked or pinned" in str(exc)
        else:
            raise AssertionError("locked entry removal should fail")

        assert cache.unlock("locked")
        assert cache.remove("locked") is None


def test_tiered_cache_tracks_nested_lock_counts(tmp_path):
    with TieredCache(dram_capacity_bytes=8, disk_path=tmp_path) as cache:
        cache.put("locked", b"aaaa")
        assert cache.lock("locked")
        assert cache.lock("locked")

        stats = cache.stats()
        assert stats.locked_entries == 1
        assert stats.lock_count == 2
        assert stats.locked_bytes == 4
        assert cache.is_resident("locked")

        assert cache.unlock("locked")
        stats = cache.stats()
        assert stats.locked_entries == 1
        assert stats.lock_count == 1
        assert stats.locked_bytes == 4

        try:
            cache.remove("locked")
        except RuntimeError as exc:
            assert "locked or pinned" in str(exc)
        else:
            raise AssertionError("locked entry removal should fail")

        assert cache.unlock("locked")
        stats = cache.stats()
        assert stats.locked_entries == 0
        assert stats.lock_count == 0
        assert stats.locked_bytes == 0
        assert cache.remove("locked") is None


def test_tiered_cache_locked_disk_read_does_not_overfill_dram(tmp_path):
    with TieredCache(dram_capacity_bytes=15, disk_path=tmp_path) as cache:
        cache.put("disk-backed", b"a" * 10)
        cache.put("resident", b"b" * 10)
        assert not cache.is_resident("disk-backed")
        assert cache.is_resident("resident")
        assert cache.lock("disk-backed")
        assert cache.lock("resident")

        assert cache.get("disk-backed") == b"a" * 10

        stats = cache.stats()
        assert stats.dram_bytes == 10
        assert not cache.is_resident("disk-backed")
        assert stats.locked_entries == 2
        assert stats.lock_count == 2

        assert cache.unlock("disk-backed")
        assert cache.unlock("resident")


def test_tiered_cache_protects_pinned_entries_from_spill(tmp_path):
    with TieredCache(dram_capacity_bytes=8, disk_path=tmp_path) as cache:
        cache.put("pinned", b"aaaa")
        cache.put("victim", b"bbbb")
        assert cache.pin("pinned")

        cache.put("new", b"cccc")
        stats = cache.stats()

        assert stats.pinned_entries == 1
        assert cache.is_resident("pinned")
        assert not cache.is_resident("victim")

        assert cache.unpin("pinned")


def test_tiered_cache_clear_preserves_locked_and_pinned_entries(tmp_path):
    with TieredCache(dram_capacity_bytes=32, disk_path=tmp_path) as cache:
        cache.put("locked", b"aaaa")
        cache.put("pinned", b"bbbb")
        cache.put("free", b"cccc")
        assert cache.lock("locked")
        assert cache.pin("pinned")

        cache.clear()

        assert cache.get("locked") == b"aaaa"
        assert cache.get("pinned") == b"bbbb"
        assert cache.get("free") is None
        stats = cache.stats()
        assert stats.total_entries == 2
        assert stats.locked_entries == 1
        assert stats.lock_count == 1
        assert stats.locked_bytes == 4
        assert stats.pinned_entries == 1

        assert cache.unlock("locked")
        assert cache.unpin("pinned")
        cache.clear()
        assert cache.stats().total_entries == 0


def test_tiered_cache_force_clear_removes_locked_and_pinned_entries(tmp_path):
    with TieredCache(dram_capacity_bytes=32, disk_path=tmp_path) as cache:
        cache.put("locked", b"aaaa")
        cache.put("pinned", b"bbbb")
        assert cache.lock("locked")
        assert cache.pin("pinned")

        cache.clear(force=True)

        stats = cache.stats()
        assert stats.total_entries == 0
        assert stats.locked_entries == 0
        assert stats.lock_count == 0
        assert stats.pinned_entries == 0
        assert cache.get("locked") is None
        assert cache.get("pinned") is None


def test_native_filesystem_l2_adapter_round_trips_python_key_filename(tmp_path):
    object_key = object_key_string(
        model_name="facebook/opt-125m",
        kv_rank=0x01020304,
        chunk_hash=bytes.fromhex("00112233445566778899aabbccddeeff"),
        cache_salt="tenant-a",
    )

    assert (
        fs_l2_filename(object_key) == "facebook-SEP-opt-125m@0x01020304@"
        "00112233445566778899aabbccddeeff@tenant-a.data"
    )

    with FileSystemL2Adapter(tmp_path) as adapter:
        assert not adapter.exists(object_key)
        adapter.put(object_key, b"payload")

        assert adapter.exists(object_key)
        assert adapter.get(object_key) == b"payload"
        assert (tmp_path / fs_l2_filename(object_key)).read_bytes() == b"payload"

        adapter.delete(object_key)
        assert not adapter.exists(object_key)
        assert adapter.get(object_key) is None

        adapter.put(object_key, b"payload-again")
        other_key = object_key_string(
            model_name="facebook/opt-125m",
            kv_rank=0x01020304,
            chunk_hash=bytes.fromhex("ffffffffffffffffffffffffffffffff"),
            cache_salt="tenant-a",
        )
        adapter.put(other_key, b"other-payload")

        adapter.clear()
        assert not adapter.exists(object_key)
        assert not adapter.exists(other_key)
        assert adapter.get(object_key) is None
        assert adapter.get(other_key) is None


def test_cxx_storage_manager_matches_store_lookup_load_bytes(tmp_path):
    manager = CxxTieredStorageManager(dram_capacity_bytes=16, disk_path=tmp_path)
    layout = MemoryLayoutDesc(shapes=[torch.Size([8])], dtypes=[torch.uint8])
    keys = [_object_key(1), _object_key(2), _object_key(3)]

    reserved = manager.reserve_write(keys, layout, "new")
    for index, key in enumerate(keys):
        reserved[key].raw_tensor.fill_(index + 1)
    manager.finish_write(keys)

    handle = manager.submit_prefetch_task(
        keys,
        layout,
        external_request_id="req-1",
    )
    assert manager.query_prefetch_lookup_hits(handle) == 3
    assert manager.query_prefetch_status(handle) == 3

    with manager.read_prefetched_results(keys) as objs:
        assert objs is not None
        assert [bytes(obj.byte_array) for obj in objs] == [
            bytes([1] * 8),
            bytes([2] * 8),
            bytes([3] * 8),
        ]
    manager.finish_read_prefetched(keys)
    manager.close()


def test_cpp_backed_mp_server_speaks_python_mp_protocol(tmp_path):
    port = _free_port()
    server, engine = run_cpp_cache_server(
        mp_config=MPServerConfig(host="127.0.0.1", port=port, chunk_size=128),
        dram_capacity_bytes=1024,
        disk_path=str(tmp_path),
        obs_config=DEFAULT_OBSERVABILITY_CONFIG,
        return_engine=True,
        start_prometheus_http_server=False,
    )
    context = zmq.Context.instance()
    client = MessageQueueClient(f"tcp://127.0.0.1:{port}", context)
    try:
        # Give the server thread a brief chance to enter poll().
        time.sleep(0.05)
        assert client.submit_request(RequestType.PING, []).result(timeout=5) is True
        assert (
            client.submit_request(RequestType.GET_CHUNK_SIZE, []).result(timeout=5)
            == 128
        )
        assert client.submit_request(RequestType.NOOP, []).result(timeout=5) == "OK"
    finally:
        client.close()
        server.close()
        engine.close()


def test_native_blake3_chunk_hashes_match_python_token_hasher():
    tokens = list(range(1, 17))
    hasher = TokenHasher(chunk_size=4, hash_algorithm="blake3")

    assert blake3_none_hash() == hasher.none_hash
    assert blake3_hash_tokens(tokens[:4]) == hasher.hash_to_bytes(
        hasher.hash_tokens(tokens[:4])
    )
    assert blake3_chunk_hashes(tokens, chunk_size=4) == hasher.compute_chunk_hashes(
        tokens
    )
    assert blake3_chunk_hashes(tokens, chunk_size=4, start=4, end=12) == (
        hasher.compute_chunk_hashes(tokens, start=4, end=12)
    )


def test_native_blake3_chunk_hashes_match_python_for_nontrivial_tokens():
    tokens = [
        0,
        1,
        255,
        256,
        65535,
        65536,
        999_999,
        2**31 - 1,
        42,
        43,
        44,
        45,
        1000,
        1001,
        1002,
        1003,
        2000,
        2001,
        2002,
    ]
    hasher = TokenHasher(chunk_size=4, hash_algorithm="blake3")

    for start, end in [
        (0, 4),
        (0, 16),
        (4, 12),
        (8, 16),
        (12, 20),
    ]:
        assert blake3_chunk_hashes(
            tokens,
            chunk_size=4,
            start=start,
            end=end,
        ) == hasher.compute_chunk_hashes(tokens, start=start, end=end)


def test_native_kv_rank_matches_python_object_key():
    assert compute_kv_rank(
        world_size=8,
        global_rank=5,
        local_world_size=4,
        local_rank=1,
    ) == ObjectKey.ComputeKVRank(
        world_size=8,
        global_rank=5,
        local_world_size=4,
        local_rank=1,
    )


def test_native_object_key_expansion_matches_python():
    chunk_hash = bytes.fromhex("00112233445566778899aabbccddeeff")
    ipc_key = IPCCacheEngineKey.from_token_ids(
        model_name="facebook/opt-125m",
        world_size=4,
        worker_id=None,
        token_ids=list(range(8)),
        request_id="req-1",
        cache_salt="tenant-a",
    )
    object_keys = ipc_key_to_object_keys(ipc_key, [chunk_hash])

    assert expand_kv_ranks(world_size=4, worker_id=None) == [
        key.kv_rank for key in object_keys
    ]
    assert [
        object_key_string(
            model_name=key.model_name,
            kv_rank=key.kv_rank,
            chunk_hash=key.chunk_hash,
            cache_salt=key.cache_salt,
        )
        for key in object_keys
    ] == [object_key_to_string(key) for key in object_keys]

    worker_key = ipc_key_to_object_keys(
        IPCCacheEngineKey.from_token_ids(
            model_name="facebook/opt-125m",
            world_size=4,
            worker_id=2,
            token_ids=list(range(8)),
            request_id="req-1",
        ),
        [chunk_hash],
    )
    assert expand_kv_ranks(world_size=4, worker_id=2) == [
        key.kv_rank for key in worker_key
    ]


def test_native_object_key_strings_match_python_for_real_chunk_hashes():
    tokens = [11, 22, 33, 44, 55, 66, 77, 88]
    hasher = TokenHasher(chunk_size=4, hash_algorithm="blake3")

    for ipc_key in [
        IPCCacheEngineKey.from_token_ids(
            model_name="facebook/opt-125m",
            world_size=3,
            worker_id=None,
            token_ids=tokens,
            request_id="req-all-ranks",
            cache_salt="tenant-a",
        ),
        IPCCacheEngineKey.from_token_ids(
            model_name="facebook/opt-125m",
            world_size=3,
            worker_id=1,
            token_ids=tokens,
            request_id="req-one-rank",
            cache_salt="",
        ),
    ]:
        chunk_hashes = hasher.compute_chunk_hashes(list(ipc_key.token_ids))
        python_keys = ipc_key_to_object_keys(ipc_key, chunk_hashes)

        native_hashes = blake3_chunk_hashes(
            ipc_key.token_ids,
            chunk_size=hasher.chunk_size,
        )
        native_ranks = expand_kv_ranks(ipc_key.world_size, ipc_key.worker_id)
        assert native_hashes == chunk_hashes
        assert [
            object_key_string(
                model_name=ipc_key.model_name,
                kv_rank=rank,
                chunk_hash=chunk_hash,
                cache_salt=ipc_key.cache_salt,
            )
            for chunk_hash in native_hashes
            for rank in native_ranks
        ] == [object_key_to_string(key) for key in python_keys]


def test_native_ipc_key_decode_matches_python_msgspec_wire():
    ipc_key = IPCCacheEngineKey.from_token_ids(
        model_name="facebook/opt-125m",
        world_size=4,
        worker_id=2,
        token_ids=[3, 1, 4, 1, 5, 9, 2, 6],
        start=4,
        end=8,
        request_id="req-native-decode",
        cache_salt="tenant-a",
    )
    decoded = decode_ipc_key(msgspec_encode(ipc_key, cls=IPCCacheEngineKey))

    assert decoded.model_name == ipc_key.model_name
    assert decoded.world_size == ipc_key.world_size
    assert decoded.worker_id == ipc_key.worker_id
    assert decoded.token_ids == ipc_key.token_ids
    assert decoded.start == ipc_key.start
    assert decoded.end == ipc_key.end
    assert decoded.request_id == ipc_key.request_id
    assert decoded.cache_salt == ipc_key.cache_salt


def test_native_ipc_key_object_key_expansion_matches_python_for_ranges():
    tokens = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]
    hasher = TokenHasher(chunk_size=4, hash_algorithm="blake3")

    for ipc_key in [
        IPCCacheEngineKey.from_token_ids(
            model_name="facebook/opt-125m",
            world_size=3,
            worker_id=None,
            token_ids=tokens,
            start=4,
            end=12,
            request_id="native-expand-all-ranks",
            cache_salt="tenant-a",
        ),
        IPCCacheEngineKey.from_token_ids(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            world_size=4,
            worker_id=2,
            token_ids=tokens,
            start=0,
            end=8,
            request_id="native-expand-worker-rank",
        ),
    ]:
        chunk_hashes = hasher.compute_chunk_hashes(
            list(ipc_key.token_ids),
            start=ipc_key.start,
            end=ipc_key.end,
        )
        assert object_key_strings(
            msgspec_encode(ipc_key, cls=IPCCacheEngineKey),
            chunk_size=hasher.chunk_size,
        ) == [
            object_key_to_string(key)
            for key in ipc_key_to_object_keys(ipc_key, chunk_hashes)
        ]

    empty_key = IPCCacheEngineKey.from_token_ids(
        model_name="facebook/opt-125m",
        world_size=2,
        worker_id=None,
        token_ids=tokens,
        start=8,
        end=8,
        request_id="native-expand-empty-range",
    )
    assert (
        object_key_strings(
            msgspec_encode(empty_key, cls=IPCCacheEngineKey),
            chunk_size=hasher.chunk_size,
        )
        == []
    )


def test_native_protocol_constants_match_python_schema():
    schema = get_protocol_schema()
    request_types = schema["request_types"]

    assert protocol_version() == LMCACHE_MP_PROTOCOL_VERSION
    assert schema["protocol_version"] == LMCACHE_MP_PROTOCOL_VERSION
    assert set(request_types) == {member.name for member in RequestType}

    for member in RequestType:
        assert request_type_value(member.name) == member.value
        assert request_type_name(member.value) == member.name
        assert request_types[member.name]["value"] == member.value
