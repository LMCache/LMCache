# SPDX-License-Identifier: Apache-2.0
"""Chunk-store plugin loading, chunk boundaries and transport contracts."""

# Standard
from dataclasses import replace
from unittest.mock import MagicMock
import socket

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.chunk_store import ChunkStoreModule
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import iter_request_handlers
from lmcache.v1.multiprocess.server_module import (
    ServerModuleSpec,
    load_server_modules,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server
from tests.v1.multiprocess.transport_test_utils import (
    REQUEST_TRANSPORTS,
    RequestTransport,
    request_server_config,
    request_server_url,
)

PLUGIN = "lmcache.v1.multiprocess.modules.chunk_store"


@pytest.fixture
def key() -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "model",
        1,
        0,
        list(range(64)),
        start=16,
        end=64,
        request_id="request",
        cache_salt="tenant",
        request_configs={"blend": True},
    )


@pytest.fixture
def plugin() -> tuple[ChunkStoreModule, MagicMock]:
    ctx = MagicMock(chunk_size=16)
    transfer = MagicMock(spec=LMCacheDrivenTransferModule)
    cache = transfer.get_and_touch_context_entry.return_value.cache_context
    cache.kv_layer_groups_manager.num_kernel_groups = 2
    cache.calculate_num_blocks.side_effect = lambda size, group: (2, 1)[group]
    transfer.store.side_effect = [
        (b"chunk-0", True),
        (b"chunk-1", True),
        (b"chunk-2", True),
    ]
    modules = load_server_modules(
        [ServerModuleSpec(PLUGIN)],
        server_context=ctx,
        mp_config=MPServerConfig(),
        coordinator_config=MagicMock(),
        built_modules=[transfer],
    )
    assert len(modules) == 1
    module = modules[0]
    assert isinstance(module, ChunkStoreModule)
    assert module.context is ctx
    return module, transfer


def test_plugin_requires_lmcache_driven_transfer() -> None:
    with pytest.raises(ValueError, match="requires LMCache-driven"):
        load_server_modules(
            [ServerModuleSpec(PLUGIN)],
            server_context=MagicMock(),
            mp_config=MPServerConfig(),
            coordinator_config=MagicMock(),
            built_modules=[],
        )


def test_plugin_is_opt_in(plugin: tuple[ChunkStoreModule, MagicMock]) -> None:
    module, transfer = plugin
    assert (
        load_server_modules(
            [],
            server_context=module.context,
            mp_config=MPServerConfig(),
            coordinator_config=MagicMock(),
            built_modules=[transfer],
        )
        == []
    )
    assert "store_with_chunk_events" not in {
        handler.operation
        for handler in iter_request_handlers(LMCacheDrivenTransferModule)
    }
    handlers = list(iter_request_handlers(module))
    assert [handler.operation for handler in handlers] == ["store_with_chunk_events"]
    assert module.report_status() == {"chunk_store": {"is_healthy": True}}
    module.close()
    transfer.close.assert_not_called()


def test_chunks_preserve_prefix_metadata_and_group_geometry(
    plugin: tuple[ChunkStoreModule, MagicMock],
    key: IPCCacheServerKey,
) -> None:
    module, transfer = plugin
    blocks = [[2, 3, 4, 5, 6, 7], [8, 9, 10]]
    assert module.store_with_chunk_events(key, 7, blocks, b"producer") == (
        b"chunk-2",
        [(b"chunk-0", 16, 32), (b"chunk-1", 32, 48), (b"chunk-2", 48, 64)],
        True,
    )
    for chunk, call in enumerate(transfer.store.call_args_list):
        chunk_key, instance, chunk_blocks, producer = call.args
        assert chunk_key == replace(key, start=16 + chunk * 16, end=32 + chunk * 16)
        assert chunk_key.request_id == key.request_id
        assert chunk_key.request_configs == key.request_configs
        assert instance == 7 and producer == b"producer"
        assert chunk_blocks == [
            blocks[0][chunk * 2 : chunk * 2 + 2],
            blocks[1][chunk : chunk + 1],
        ]
        # The ordinary transfer may downsample its input lists in place.
        chunk_blocks[0].clear()
    assert blocks == [[2, 3, 4, 5, 6, 7], [8, 9, 10]]


@pytest.mark.parametrize("start,end", [(-16, 32), (32, 16), (0, 80), (1, 32), (0, 17)])
def test_invalid_range_submits_nothing(
    plugin: tuple[ChunkStoreModule, MagicMock],
    key: IPCCacheServerKey,
    start: int,
    end: int,
) -> None:
    module, transfer = plugin
    assert module.store_with_chunk_events(
        replace(key, start=start, end=end), 7, [[1, 2], [3]], b"producer"
    ) == (b"", [], False)
    transfer.store.assert_not_called()


@pytest.mark.parametrize(
    "blocks", [[], [[1] * 6], [[1] * 6, [2] * 2], [[1] * 7, [2] * 3]]
)
def test_invalid_blocks_reject_whole_request(
    plugin: tuple[ChunkStoreModule, MagicMock],
    key: IPCCacheServerKey,
    blocks: list[list[int]],
) -> None:
    module, transfer = plugin
    assert module.store_with_chunk_events(key, 7, blocks, b"producer") == (
        b"",
        [],
        False,
    )
    transfer.store.assert_not_called()


def test_empty_range_is_a_noop(
    plugin: tuple[ChunkStoreModule, MagicMock],
    key: IPCCacheServerKey,
) -> None:
    module, transfer = plugin
    assert module.store_with_chunk_events(
        replace(key, end=key.start), 7, [[], []], b"producer"
    ) == (b"", [], True)
    transfer.store.assert_not_called()


def test_unregistered_worker_submits_nothing(
    plugin: tuple[ChunkStoreModule, MagicMock],
    key: IPCCacheServerKey,
) -> None:
    module, transfer = plugin
    transfer.get_and_touch_context_entry.return_value = None
    assert module.store_with_chunk_events(key, 7, [[1] * 6, [2] * 3], b"producer") == (
        b"",
        [],
        False,
    )
    transfer.store.assert_not_called()


@pytest.mark.parametrize("failed_handle", [b"failed-chunk", b""])
def test_failed_chunk_preserves_completion_and_stops_submission(
    plugin: tuple[ChunkStoreModule, MagicMock],
    key: IPCCacheServerKey,
    failed_handle: bytes,
) -> None:
    module, transfer = plugin
    transfer.store.side_effect = [(b"chunk-0", True), (failed_handle, False)]
    expected_chunks = [(b"chunk-0", 16, 32)]
    if failed_handle:
        expected_chunks.append((failed_handle, 32, 48))
    assert module.store_with_chunk_events(key, 7, [[1] * 6, [2] * 3], b"producer") == (
        failed_handle or b"chunk-0",
        expected_chunks,
        False,
    )
    assert transfer.store.call_count == 2


@pytest.mark.parametrize("transport", REQUEST_TRANSPORTS)
def test_loaded_plugin_rpc_round_trip(
    plugin: tuple[ChunkStoreModule, MagicMock],
    key: IPCCacheServerKey,
    transport: RequestTransport,
) -> None:
    module, transfer = plugin
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    url = request_server_url(transport, port)
    server = create_request_server([module], request_server_config(transport, url))
    server.start()
    client = RequestClientFactory.create(url)
    try:
        assert client.store_with_chunk_events(
            key, 7, [[1] * 6, [2] * 3], b"producer"
        ).result(10) == (
            b"chunk-2",
            [(b"chunk-0", 16, 32), (b"chunk-1", 32, 48), (b"chunk-2", 48, 64)],
            True,
        )
        assert transfer.store.call_count == 3
        transfer.get_and_touch_context_entry.return_value = None
        assert client.store_with_chunk_events(
            key, 7, [[1] * 6, [2] * 3], b"producer"
        ).result(10) == (b"", [], False)
    finally:
        client.close()
        server.close()
