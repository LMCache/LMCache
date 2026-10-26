# SPDX-License-Identifier: Apache-2.0
"""MooncakestoreConnector: Mooncake return codes are not treated as success.

``MooncakeDistributedStore`` reports failures through return codes. A fake
``mooncake.store`` module stands in for it, so no Mooncake build is needed.
"""

# Standard
from typing import Any, Iterator
import asyncio
import logging
import sys
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.ad_hoc_memory_allocator import (
    AdHocMemoryAllocator,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
    MooncakestoreConnector,
)
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

# Mooncake ErrorCode values seen in kvcache-ai/Mooncake#4625.
NO_SPACE = -200
CLIENT_NOT_INITIALIZED = -600


class FakeStore:
    """Dict-API ``MooncakeDistributedStore`` returning configured codes."""

    setup_ret = 0

    def __init__(self) -> None:
        self.exist_codes: dict[str, int] = {}
        self.put_code = 0

    def setup(self, config: dict) -> int:
        return FakeStore.setup_ret

    def get_hostname(self) -> str:
        return "localhost"

    def is_exist(self, key: str) -> int:
        return self.exist_codes.get(key, 0)

    def put_from(self, key: str, ptr: int, size: int, config: Any) -> int:
        return self.put_code

    def batch_put_from(
        self, keys: list[str], ptrs: list[int], sizes: list[int], config: Any
    ) -> list[int]:
        return [self.put_code] * len(keys)

    def put_parts(self, key: str, *parts: Any) -> int:
        return self.put_code

    def close(self) -> None:
        pass


class FakeLegacyStore(FakeStore):
    """Store with only the legacy positional ``setup(...)`` API."""

    def setup(self, *args: Any) -> int:  # type: ignore[override]
        if len(args) == 1 and isinstance(args[0], dict):
            raise TypeError("setup(): incompatible function arguments")
        return FakeStore.setup_ret


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _install_fake_mooncake(
    monkeypatch: pytest.MonkeyPatch, store_cls: type[FakeStore]
) -> None:
    store_mod = types.ModuleType("mooncake.store")
    store_mod.MooncakeDistributedStore = store_cls  # type: ignore[attr-defined]
    store_mod.ReplicateConfig = types.SimpleNamespace  # type: ignore[attr-defined]
    pkg = types.ModuleType("mooncake")
    pkg.store = store_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", pkg)
    monkeypatch.setitem(sys.modules, "mooncake.store", store_mod)


def _make_connector(
    save_chunk_meta: bool,
) -> tuple[MooncakestoreConnector, LocalCPUBackend]:
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=256,
        local_cpu=True,
        max_local_cpu_size=1.0,
        remote_serde="naive",
        extra_config={
            "save_chunk_meta": save_chunk_meta,
            "local_hostname": "127.0.0.1",
            "metadata_server": "P2PHANDSHAKE",
            "master_server_address": "127.0.0.1:50051",
        },
    )
    metadata = LMCacheMetadata(
        model_name="m",
        kv_dtype=torch.float16,
        kv_shape=(2, 2, 256, 4, 64),
        use_mla=False,
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
    )
    local_cpu = LocalCPUBackend(
        config, metadata, memory_allocator=AdHocMemoryAllocator(device="cpu")
    )
    connector = MooncakestoreConnector(asyncio.new_event_loop(), local_cpu, config)
    return connector, local_cpu


def _keys(n: int) -> list[CacheEngineKey]:
    return [CacheEngineKey("m", 1, 0, 1000 + i, torch.float16) for i in range(n)]


def _memory_objs(local_cpu: LocalCPUBackend, n: int) -> list[MemoryObj]:
    objs = []
    for _ in range(n):
        obj = local_cpu.allocate(torch.Size([2, 2, 256, 256]), torch.float16)
        assert obj is not None
        objs.append(obj)
    return objs


@pytest.fixture
def fake_mooncake(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    _install_fake_mooncake(monkeypatch, FakeStore)
    FakeStore.setup_ret = 0
    yield
    FakeStore.setup_ret = 0


@pytest.fixture
def connector_warnings() -> Iterator[list[str]]:
    handler = ListHandler()
    logger = logging.getLogger(MooncakestoreConnector.__module__)
    logger.addHandler(handler)
    yield handler.messages
    logger.removeHandler(handler)


@pytest.mark.parametrize("store_cls", [FakeStore, FakeLegacyStore])
def test_setup_error_code_fails_connector_creation(
    monkeypatch: pytest.MonkeyPatch, store_cls: type[FakeStore]
) -> None:
    _install_fake_mooncake(monkeypatch, store_cls)
    monkeypatch.setattr(FakeStore, "setup_ret", CLIENT_NOT_INITIALIZED)

    with pytest.raises(RuntimeError, match=str(CLIENT_NOT_INITIALIZED)):
        _make_connector(save_chunk_meta=False)


@pytest.mark.parametrize("store_cls", [FakeStore, FakeLegacyStore])
def test_setup_success_creates_connector(
    monkeypatch: pytest.MonkeyPatch, store_cls: type[FakeStore]
) -> None:
    _install_fake_mooncake(monkeypatch, store_cls)

    connector, _ = _make_connector(save_chunk_meta=False)

    assert isinstance(connector.store, store_cls)


@pytest.mark.usefixtures("fake_mooncake")
def test_exists_counts_only_one_as_hit(connector_warnings: list[str]) -> None:
    connector, _ = _make_connector(save_chunk_meta=False)
    hit, miss, error = _keys(3)
    connector.store.exist_codes = {
        hit.to_string(): 1,
        miss.to_string(): 0,
        error.to_string(): CLIENT_NOT_INITIALIZED,
    }

    assert connector.exists_sync(hit) is True
    assert connector.exists_sync(miss) is False
    assert connector.exists_sync(error) is False
    assert asyncio.run(connector.exists(error)) is False
    assert any(str(CLIENT_NOT_INITIALIZED) in m for m in connector_warnings)


@pytest.mark.usefixtures("fake_mooncake")
def test_batched_async_contains_stops_at_error_code() -> None:
    connector, _ = _make_connector(save_chunk_meta=False)
    keys = _keys(3)
    connector.store.exist_codes = {
        keys[0].to_string(): 1,
        keys[1].to_string(): CLIENT_NOT_INITIALIZED,
        keys[2].to_string(): 1,
    }

    assert asyncio.run(connector.batched_async_contains("lookup", keys)) == 1


@pytest.mark.usefixtures("fake_mooncake")
@pytest.mark.parametrize("save_chunk_meta", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_failed_put_logs_key_and_code(
    connector_warnings: list[str], save_chunk_meta: bool, batched: bool
) -> None:
    connector, local_cpu = _make_connector(save_chunk_meta=save_chunk_meta)
    connector.store.put_code = NO_SPACE
    keys = _keys(2)
    objs = _memory_objs(local_cpu, 2)

    if batched:
        asyncio.run(connector.batched_put(keys, objs))
    else:
        asyncio.run(connector.put(keys[0], objs[0]))

    failed_keys = keys if batched else keys[:1]
    for key in failed_keys:
        assert any(
            key.to_string() in m and str(NO_SPACE) in m for m in connector_warnings
        )


@pytest.mark.usefixtures("fake_mooncake")
@pytest.mark.parametrize("save_chunk_meta", [False, True])
def test_successful_put_logs_nothing(
    connector_warnings: list[str], save_chunk_meta: bool
) -> None:
    connector, local_cpu = _make_connector(save_chunk_meta=save_chunk_meta)
    keys = _keys(2)
    objs = _memory_objs(local_cpu, 2)

    asyncio.run(connector.batched_put(keys, objs))
    asyncio.run(connector.put(keys[0], objs[0]))

    assert connector_warnings == []
