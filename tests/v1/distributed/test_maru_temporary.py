# SPDX-License-Identifier: Apache-2.0
"""Temporary L1 pages stay private through completion and cleanup."""

# Standard
from collections.abc import Iterator
from unittest.mock import MagicMock
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.maru_l1_manager import (
    MaruL1Manager,
    object_key_to_string,
)
from lmcache.v1.mp_observability.event import EventType

# Local
from .maru_fakes import FakeCxlAdapter, FakeMaruHandler, make_maru_manager

_LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([4, 8])], dtypes=[torch.float16])


def _key(index: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([index]), model_name="temporary", kv_rank=0)


@pytest.fixture
def local_manager() -> Iterator[tuple[MaruL1Manager, FakeMaruHandler, FakeCxlAdapter]]:
    """Provide a manager whose directory and page releases are observable."""
    manager, handler, adapter = make_maru_manager()
    try:
        yield manager, handler, adapter
    finally:
        manager.close()


def test_temporary_finish_write_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A peer cannot pin a completed temporary, and no store event is sent."""
    bus = MagicMock()
    monkeypatch.setattr(
        "lmcache.v1.distributed.maru_l1_manager.get_event_bus", lambda: bus
    )
    manager, handler, adapter = make_maru_manager()
    key = _key(1)
    try:
        _, obj = manager.reserve_write([key], [True], _LAYOUT, mode="new")[key]
        assert obj is not None
        assert manager.finish_write([key])[key] == L1Error.SUCCESS
        assert handler.batch_exists([object_key_to_string(key)]) == [False]
        assert handler.batch_pin([object_key_to_string(key)]) == [False]
        assert not any(
            call.args[0].event_type == EventType.L1_WRITE_FINISHED
            for call in bus.publish.call_args_list
        )
        assert adapter.freed == []
        assert manager.delete([key])[key] == L1Error.SUCCESS
        assert adapter.freed == [obj.metadata.address]
    finally:
        manager.close()


@pytest.mark.parametrize("promote", [False, True])
def test_private_read_and_cleanup_do_not_touch_peer_copy(
    local_manager: tuple[MaruL1Manager, FakeMaruHandler, FakeCxlAdapter],
    promote: bool,
) -> None:
    """A peer publishing the same key cannot change a private page's lifetime."""
    manager, handler, adapter = local_manager
    key = _key(2)
    encoded = object_key_to_string(key)
    _, obj = manager.reserve_write([key], [True], _LAYOUT, mode="new")[key]
    assert obj is not None
    if promote:
        assert manager.finish_write_and_reserve_read([key])[key] == (
            L1Error.SUCCESS,
            obj,
        )
    else:
        assert manager.finish_write([key])[key] == L1Error.SUCCESS
    # Simulate a peer's independently allocated copy of the same logical key.
    handler.store_map[encoded] = (7, 3, 64)
    handler.pins[encoded] = 2
    assert manager.reserve_read([key])[key] == (L1Error.SUCCESS, obj)
    assert handler.pins[encoded] == 2
    assert manager.finish_read([key])[key] == L1Error.SUCCESS
    if promote:
        assert adapter.freed == []
        assert manager.finish_read([key])[key] == L1Error.SUCCESS
    assert adapter.freed == [obj.metadata.address]
    assert handler.store_map[encoded] == (7, 3, 64)
    assert handler.pins[encoded] == 2
    assert handler.unpin_log == []


def test_deleting_completed_temporary_preserves_peer_copy(
    local_manager: tuple[MaruL1Manager, FakeMaruHandler, FakeCxlAdapter],
) -> None:
    """Finish/delete cleanup reclaims only the local unregistered allocation."""
    manager, handler, adapter = local_manager
    key = _key(3)
    encoded = object_key_to_string(key)
    _, obj = manager.reserve_write([key], [True], _LAYOUT, mode="new")[key]
    assert obj is not None
    assert manager.finish_write([key])[key] == L1Error.SUCCESS
    handler.store_map[encoded] = (7, 3, 64)
    handler.pins[encoded] = 1
    assert manager.delete([key])[key] == L1Error.SUCCESS
    assert adapter.freed == [obj.metadata.address]
    assert handler.store_map[encoded] == (7, 3, 64)
    assert handler.pins[encoded] == 1


def test_mixed_finish_write_keeps_temporary_private_on_store_failure(
    local_manager: tuple[MaruL1Manager, FakeMaruHandler, FakeCxlAdapter],
) -> None:
    """Directory store failure does not discard or publish a private buffer."""
    manager, handler, adapter = local_manager
    temporary, retained = _key(4), _key(5)
    reserved = manager.reserve_write(
        [temporary, retained], [True, False], _LAYOUT, mode="new"
    )
    temporary_obj = reserved[temporary][1]
    assert temporary_obj is not None
    handler.fail_store = True
    result = manager.finish_write([temporary, retained])
    assert result[temporary] == L1Error.SUCCESS
    assert result[retained] == L1Error.KEY_IN_WRONG_STATE
    assert handler.batch_exists([object_key_to_string(temporary)]) == [False]
    assert manager.reserve_read([temporary])[temporary] == reserved[temporary]
    assert manager.finish_read([temporary])[temporary] == L1Error.SUCCESS
    assert adapter.freed == [temporary_obj.metadata.address]


def test_non_force_clear_keeps_active_temporary_read(
    local_manager: tuple[MaruL1Manager, FakeMaruHandler, FakeCxlAdapter],
) -> None:
    """Clear reclaims completed idle pages but preserves a reader's buffer."""
    manager, _, adapter = local_manager
    idle, active = _key(6), _key(7)
    reserved = manager.reserve_write([idle, active], [True, True], _LAYOUT, "new")
    idle_obj = reserved[idle][1]
    assert idle_obj is not None
    assert all(
        v == L1Error.SUCCESS for v in manager.finish_write([idle, active]).values()
    )
    assert manager.reserve_read([active])[active] == reserved[active]
    manager.clear()
    assert manager.unsafe_read([active])[active] == reserved[active]
    assert manager.reserve_read([idle])[idle] == (L1Error.KEY_NOT_EXIST, None)
    assert adapter.freed == [idle_obj.metadata.address]
    assert manager.finish_read([active])[active] == L1Error.SUCCESS


def test_close_reclaims_completed_temporary_once() -> None:
    """Shutdown must not leak or double-free a finished, unread temporary."""
    manager, _, adapter = make_maru_manager()
    key = _key(8)
    _, obj = manager.reserve_write([key], [True], _LAYOUT, mode="new")[key]
    assert obj is not None
    assert manager.finish_write([key])[key] == L1Error.SUCCESS
    manager.close()
    assert adapter.freed == [obj.metadata.address]


def test_completed_temporary_without_reader_is_not_released_by_finish_read(
    local_manager: tuple[MaruL1Manager, FakeMaruHandler, FakeCxlAdapter],
) -> None:
    """An unmatched finish_read must not free an unlocked temporary page."""
    manager, _, adapter = local_manager
    key = _key(9)
    manager.reserve_write([key], [True], _LAYOUT, mode="new")
    assert manager.finish_write([key])[key] == L1Error.SUCCESS
    assert manager.finish_read([key])[key] == L1Error.KEY_IN_WRONG_STATE
    assert adapter.freed == []
    assert manager.delete([key])[key] == L1Error.SUCCESS


def test_completed_temporary_is_reclaimed_after_read_ttl() -> None:
    """An abandoned unlocked temporary is reclaimed by the public lifecycle."""
    manager, handler, adapter = make_maru_manager(read_ttl_seconds=1)
    key = _key(10)
    try:
        _, obj = manager.reserve_write([key], [True], _LAYOUT, mode="new")[key]
        assert obj is not None
        assert manager.finish_write([key])[key] == L1Error.SUCCESS
        deadline = time.monotonic() + 5
        while not adapter.freed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert adapter.freed == [obj.metadata.address]
        assert manager.reserve_read([key])[key] == (L1Error.KEY_NOT_EXIST, None)
        assert handler.unpin_log == []
    finally:
        manager.close()
