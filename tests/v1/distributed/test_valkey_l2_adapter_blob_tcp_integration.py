# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for the ``valkey`` adapter's ``blob-tcp`` transport.

They run against a real server with the valkey-large-object module loaded and
a stock ``valkey-glide-sync`` (no RDMA build or libfabric needed). Skipped
unless ``LMCACHE_VALKEY_BLOB_SERVER=<host>:<port>`` names such a server.

The server must accept the 4 MiB chunk these tests store. The module requires
``max-object-size`` to fit in one segment after its per-chunk overhead, which
with 4 KiB chunks is about half the segment, so this command allows objects
up to about 8 MiB. ``blob-tcp`` needs no fabric; the last two arguments only
let the same server also run the RDMA integration tests::

    valkey-server --port 6379 \\
      --loadmodule <ValkeyLargeObj>/target/release/libvalkey_large_object.so \\
        operating-mode Dram segment-size 16777216 max-object-size 8384512 \\
        chunk-size 4096 fabric-provider Emulated fabric-interfaces lo

    LMCACHE_VALKEY_BLOB_SERVER=127.0.0.1:6379 pytest <thisfile> -v
"""

# Standard
from typing import Any, Callable
import os
import select
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
)
from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (
    ValkeyL2Adapter,
    ValkeyL2AdapterConfig,
)
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.platform import consume_fd

SERVER_ENV = "LMCACHE_VALKEY_BLOB_SERVER"

_EMPTY_LAYOUT = MemoryLayoutDesc(shapes=[], dtypes=[])
CHUNK_BYTES = 4 << 20


def _real_glide() -> bool:
    """Whether the installed ``glide_sync`` is the real package, not a fake."""
    try:
        # Third Party
        import glide_sync
    except ImportError:
        return False
    return getattr(glide_sync, "__spec__", None) is not None


pytestmark = [
    pytest.mark.skipif(
        not os.getenv(SERVER_ENV),
        reason=f"set {SERVER_ENV}=<host>:<port> to a server running the "
        "valkey-large-object module",
    ),
    pytest.mark.skipif(not _real_glide(), reason="needs valkey-glide-sync"),
]


def _obj(nbytes: int = CHUNK_BYTES, fill: float = 1.0) -> TensorMemoryObj:
    raw = torch.empty(nbytes // 4, dtype=torch.float32)
    raw.fill_(fill)
    meta = MemoryObjMetadata(
        shape=torch.Size([nbytes // 4]),
        dtype=torch.float32,
        address=0,
        phy_size=nbytes,
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(raw, meta, parent_allocator=None)


def _bytes_of(obj: TensorMemoryObj) -> bytes:
    return bytes(obj.byte_array)


def _key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name=f"blob_tcp_it_{os.getpid()}",
        kv_rank=0,
        cache_salt="",
    )


def _wait(fd: int, ready: Callable[[], Any], timeout: float = 30.0) -> Any:
    """Poll ``fd`` until ``ready()`` returns something other than None."""
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not poll.poll(remaining * 1000):
            break
        try:
            consume_fd(fd)
        except BlockingIOError:
            pass
        result = ready()
        if result is not None:
            return result
    raise AssertionError(f"no completion within {timeout}s")


def _store(adapter: ValkeyL2Adapter, keys: list, objs: list) -> Any:
    task_id = adapter.submit_store_task(keys, objs)
    return _wait(
        adapter.get_store_event_fd(),
        lambda: adapter.pop_completed_store_tasks().get(task_id),
    )


def _load(adapter: ValkeyL2Adapter, keys: list, objs: list) -> Any:
    task_id = adapter.submit_load_task(keys, objs)
    return _wait(
        adapter.get_load_event_fd(), lambda: adapter.query_load_result(task_id)
    )


def _lookup(adapter: ValkeyL2Adapter, keys: list) -> Any:
    task_id = adapter.submit_lookup_and_lock_task(keys, {0: _EMPTY_LAYOUT})
    return _wait(
        adapter.get_lookup_and_lock_event_fd(),
        lambda: adapter.query_lookup_and_lock_result(task_id),
    )


def _make_adapter(**overrides: Any) -> ValkeyL2Adapter:
    fields: dict[str, Any] = {
        "type": "valkey",
        "transport": "blob-tcp",
        "startup_nodes": os.environ[SERVER_ENV],
        "num_workers": 2,
        "connection_timeout": 10.0,
        "request_timeout": 10.0,
    }
    fields.update(overrides)
    adapter = create_l2_adapter_from_registry(
        ValkeyL2AdapterConfig.from_dict(fields), None
    )
    assert isinstance(adapter, ValkeyL2Adapter)
    return adapter


@pytest.fixture
def adapter():
    adapter = _make_adapter()
    yield adapter
    adapter.close()


class TestBlobTcpTransfers:
    def test_a_4mib_chunk_round_trips(self, adapter: ValkeyL2Adapter):
        key = _key(1)
        src = _obj(fill=3.0)
        try:
            result = _store(adapter, [key], [src])
            assert result.is_successful()
            assert result.bytes_transferred() == CHUNK_BYTES

            dst = _obj(fill=0.0)
            assert _load(adapter, [key], [dst]).test(0)
            assert _bytes_of(dst) == _bytes_of(src)
        finally:
            adapter.delete([key])

    def test_a_missing_key_leaves_the_destination_alone(self, adapter: ValkeyL2Adapter):
        dst = _obj(fill=7.0)
        before = _bytes_of(dst)

        assert not _load(adapter, [_key(2)], [dst]).test(0)
        assert _bytes_of(dst) == before

    def test_ttl_is_applied(self):
        # Long enough that the lookup right after the store cannot miss.
        adapter = _make_adapter(num_workers=1, ttl_seconds=3)
        key = _key(3)
        try:
            assert _store(adapter, [key], [_obj(64 << 10)]).is_successful()
            assert _lookup(adapter, [key]).test(0)
            adapter.submit_unlock([key])
            time.sleep(4.5)
            assert not _lookup(adapter, [key]).test(0)
        finally:
            adapter.delete([key])
            adapter.close()

    def test_status_reports_the_transport(self, adapter: ValkeyL2Adapter):
        status = adapter.report_status()
        assert status["is_healthy"]
        assert status["transport"] == "blob-tcp"
