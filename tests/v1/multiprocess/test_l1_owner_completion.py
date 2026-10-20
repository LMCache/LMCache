# SPDX-License-Identifier: Apache-2.0
"""Owner-tagged STORE callbacks use the production MessagePack dispatcher."""

# Standard
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import MagicMock

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import L1WriteCompletion, StorageManager
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess import native_completion
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer


@pytest.mark.parametrize("owner_count", [1, 2])
@pytest.mark.parametrize("failure", ["none", "copy", "owner"])
def test_store_owner_callback_round_trip(
    monkeypatch: pytest.MonkeyPatch,
    owner_count: int,
    failure: Literal["none", "copy", "owner"],
) -> None:
    """Real reservations survive serialization and finish only after dispatch.

    Only pinned-memory and device calls are replaced; this is a CPU contract
    test, not a CUDA ordering test. Existing native tests cover device ordering.
    """
    monkeypatch.setattr(
        "lmcache.v1.memory_management._allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(
        "lmcache.v1.memory_management._free_cpu_memory", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "lmcache.v1.memory_allocators.mixed_memory_allocator.torch_dev",
        SimpleNamespace(is_available=lambda: False),
    )
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=8192 // owner_count,
            use_lazy=False,
            align_bytes=64,
            shm_name="",
        )
    )
    managers = tuple(
        L1Manager(replace(config, tag=f"l1-{index}")) for index in range(owner_count)
    )
    storage = StorageManager(
        StorageManagerConfig(
            l1_manager_config=config,
            eviction_config=EvictionConfig(eviction_policy="noop"),
            l2_adapter_config=L2AdaptersConfig(adapters=[]),
        ),
        _l1_managers=managers,
    )
    queued: list[tuple[str, bytes]] = []
    order: list[str] = []

    def record(stream_ptr: int, kind: str, payload: bytes) -> None:
        assert stream_ptr == 123
        order.append("callback")
        queued.append((kind, payload))

    def drain() -> list[tuple[str, bytes]]:
        result = list(queued)
        queued.clear()
        return result

    monkeypatch.setattr(
        native_completion,
        "_device_ops",
        SimpleNamespace(
            record_completion_on_stream=record, drain_recorded_completions=drain
        ),
    )
    # close() performs a synchronous drain through the registered decoder.
    monkeypatch.setattr(
        native_completion.DeviceHostFuncDispatcher, "start", lambda self: None
    )
    monkeypatch.setattr(transfer, "torch_dev", MagicMock())
    monkeypatch.setattr(transfer, "downsample_and_stage_block_ids", lambda cc, ids: ids)
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    monkeypatch.setattr(transfer, "get_layout_desc", lambda *args, **kwargs: layout)
    copied: list[MemoryObj] = []

    def copy(*args: Any, **kwargs: Any) -> None:
        order.append("copy")
        copied.extend(args[2])
        if failure == "copy" and len(copied) == 2:
            raise RuntimeError("device copy failed")
        if failure == "owner" and len(copied) == 2:
            copied[-1].reset_l1_manager()

    monkeypatch.setattr(transfer, "transfer_kv_per_object_group", copy)
    keys = [ObjectKey(b"chunk", "model", 0, gid) for gid in range(2)]
    ctx = SimpleNamespace(
        chunk_size=1,
        null_block_id=0,
        storage_manager=storage,
        event_bus=MagicMock(),
        resolve_obj_keys=lambda key, gids: [[k] for k in keys],
    )
    module = transfer.LMCacheDrivenTransferModule(cast(Any, ctx))
    backend = MagicMock()
    backend.record_event.side_effect = lambda *args: order.append("record")
    cache_context = MagicMock()
    cache_context.cupy_stream.ptr = 123
    cache_context.calculate_num_blocks.return_value = 1
    cache_context.hold_imported_event.return_value = 7
    cache_context.kv_layer_groups_manager = SimpleNamespace(
        num_object_groups=2,
        num_kernel_groups=2,
        object_groups=[ObjectGroupInfo(kernel_group_indices=[i]) for i in range(2)],
    )
    entry = SimpleNamespace(
        cache_context=cache_context, model_name="model", event_backend=backend
    )
    monkeypatch.setattr(module, "get_and_touch_context_entry", lambda _: entry)
    try:
        _, succeeded = module.store(
            cast(Any, SimpleNamespace(request_id="req", worker_id=1)),
            1,
            [[1], [2]],
            b"producer",
        )
        assert succeeded is (failure == "none")
        assert len(copied) == 2
        end_event = ctx.event_bus.publish_on_stream.call_args.args[1]
        assert end_event.event_type == EventType.MP_STORE_END
        assert end_event.metadata["stored_count"] == (2 if succeeded else 0)
        if not succeeded:
            assert end_event.metadata["total_bytes"] == 0
            assert end_event.metadata["num_tokens"] == 0
        for manager in managers:
            assert all(
                err == L1Error.KEY_NOT_EXIST
                for err, _ in manager.reserve_read(keys).values()
            )
        # The imported producer event is released by a callback queued right
        # behind the stream wait, before any copy.
        release_kind, release_encoded = queued[0]
        assert release_kind == "release_imported_event"
        assert msgspec.msgpack.decode(release_encoded, type=tuple[int, int]) == (1, 7)
        if failure != "none":
            # A failed store aborts every reservation once its copies drain,
            # routed by the owner each reservation had before any copy ran.
            assert order == ["callback", "copy", "copy", "record", "callback"]
            kind, encoded = queued[1]
            assert kind == "abort_write_by_owner"
            payload = msgspec.msgpack.decode(encoded, type=L1WriteCompletion)
            assert {(owner, key) for owner, group in payload for key in group} == {
                (managers[index if owner_count == 2 else 0].l1_manager_id, key)
                for index, key in enumerate(keys)
            }
            assert all(obj.is_valid() for obj in copied)
        else:
            assert order == ["callback", "copy", "copy", "record", "callback"]
            kind, encoded = queued[1]
            assert kind == "finish_write_by_owner"
            payload = msgspec.msgpack.decode(encoded, type=L1WriteCompletion)
            assert payload == [
                (manager.l1_manager_id, keys if owner_count == 1 else [keys[index]])
                for index, manager in enumerate(managers)
            ]
        module.close()
        cache_context.release_imported_event.assert_called_once_with(7)
        # close() drained the queue: an aborted store left no staging behind.
        assert all(
            manager.report_status()["staging_object_count"] == 0 for manager in managers
        )
        for index, manager in enumerate(managers):
            found = manager.reserve_read(keys)
            expected = (
                [] if failure != "none" else keys if owner_count == 1 else [keys[index]]
            )
            assert [
                key for key, (err, _) in found.items() if err == L1Error.SUCCESS
            ] == expected
            if expected:
                manager.finish_read(expected)
    finally:
        module.close()
        storage.close()
