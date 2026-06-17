# SPDX-License-Identifier: Apache-2.0
"""
Tests for the P2P protocol and P2PController: enum registration, protocol
definitions, MemoryLayoutDesc wire serialization, and server handlers.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.multiprocess.modules.p2p_controller import P2PController
from lmcache.v1.multiprocess.mq import msgspec_decode, msgspec_encode
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType


def _make_key(i: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=f"hash{i}".encode(),
        model_name="test_model",
        kv_rank=1,
    )


def _make_layout_desc() -> MemoryLayoutDesc:
    return MemoryLayoutDesc(
        shapes=[torch.Size([2, 3]), torch.Size([4])],
        dtypes=[torch.float16, torch.bfloat16],
    )


# ============================================================================
# Protocol definition tests
# ============================================================================


def test_p2p_request_types_registered():
    """The three P2P request types should be members of RequestType."""
    for name in (
        "P2P_LOOKUP_AND_LOCK",
        "P2P_QUERY_LOOKUP_RESULTS",
        "P2P_UNLOCK_OBJECTS",
    ):
        assert hasattr(RequestType, name)
        assert isinstance(getattr(RequestType, name), RequestType)


def test_p2p_lookup_and_lock_protocol():
    """P2P_LOOKUP_AND_LOCK payload is [list[ObjectKey], MemoryLayoutDesc],
    returns int, and is BLOCKING."""
    payload_classes = get_payload_classes(RequestType.P2P_LOOKUP_AND_LOCK)
    assert payload_classes == [list[ObjectKey], MemoryLayoutDesc]
    assert get_response_class(RequestType.P2P_LOOKUP_AND_LOCK) is int
    assert get_handler_type(RequestType.P2P_LOOKUP_AND_LOCK) == HandlerType.BLOCKING


def test_p2p_query_lookup_results_protocol():
    """P2P_QUERY_LOOKUP_RESULTS payload is [int], returns the optional address
    list, and is BLOCKING."""
    assert get_payload_classes(RequestType.P2P_QUERY_LOOKUP_RESULTS) == [int]
    assert (
        get_response_class(RequestType.P2P_QUERY_LOOKUP_RESULTS)
        == list[TransferChannelAddress] | None
    )
    assert (
        get_handler_type(RequestType.P2P_QUERY_LOOKUP_RESULTS) == HandlerType.BLOCKING
    )


def test_p2p_unlock_objects_protocol():
    """P2P_UNLOCK_OBJECTS payload is [list[ObjectKey]], returns None, BLOCKING."""
    assert get_payload_classes(RequestType.P2P_UNLOCK_OBJECTS) == [list[ObjectKey]]
    assert get_response_class(RequestType.P2P_UNLOCK_OBJECTS) is None
    assert get_handler_type(RequestType.P2P_UNLOCK_OBJECTS) == HandlerType.BLOCKING


# ============================================================================
# MemoryLayoutDesc serialization tests
# ============================================================================


def test_memory_layout_desc_mq_roundtrip():
    """The mq encode/decode dispatch must round-trip MemoryLayoutDesc, whose
    torch.Size / torch.dtype fields ride the customized enc_hook / dec_hook."""
    desc = MemoryLayoutDesc(
        shapes=[torch.Size([8, 16, 128]), torch.Size([4])],
        dtypes=[torch.float32, torch.int8],
    )
    decoded = msgspec_decode(
        msgspec_encode(desc, cls=MemoryLayoutDesc), cls=MemoryLayoutDesc
    )
    assert decoded == desc
    assert all(isinstance(s, torch.Size) for s in decoded.shapes)
    assert all(isinstance(d, torch.dtype) for d in decoded.dtypes)


def test_memory_layout_desc_empty_mq_roundtrip():
    """An empty layout descriptor must round-trip through the mq dispatch."""
    desc = MemoryLayoutDesc(shapes=[], dtypes=[])
    decoded = msgspec_decode(
        msgspec_encode(desc, cls=MemoryLayoutDesc), cls=MemoryLayoutDesc
    )
    assert decoded == desc


# ============================================================================
# TransferChannelAddress tests
# ============================================================================


def test_transfer_channel_address_validity():
    """A non-negative offset is valid; a negative one is not."""
    assert TransferChannelAddress(offset=128, size=64).is_valid()
    assert not TransferChannelAddress(offset=-1, size=0).is_valid()


# ============================================================================
# Server handler tests
# ============================================================================


def _make_controller() -> tuple[P2PController, MagicMock]:
    ctx = MagicMock()
    controller = P2PController(ctx)
    return controller, ctx


def test_lookup_and_lock_submits_skip_l2_and_returns_task_id():
    """p2p_lookup_and_lock submits a skip_l2 prefetch and returns a fresh id."""
    controller, ctx = _make_controller()
    handle = MagicMock(l1_found_indices=(0, 1))
    ctx.storage_manager.submit_prefetch_task.return_value = handle

    keys = [_make_key(0), _make_key(1)]
    layout_desc = _make_layout_desc()
    task_id = controller.p2p_lookup_and_lock(keys, layout_desc)

    assert task_id == 0
    args, kwargs = ctx.storage_manager.submit_prefetch_task.call_args
    assert args[0] == keys
    assert args[1] is layout_desc
    assert kwargs["skip_l2"] is True
    # A second call gets a distinct id.
    assert controller.p2p_lookup_and_lock(keys, layout_desc) == 1


def test_query_lookup_results_builds_addresses_for_prefix():
    """A completed lookup returns one address per key: real offsets for the
    found prefix, invalid offsets for the rest."""
    controller, ctx = _make_controller()
    handle = MagicMock(l1_found_indices=(0, 1))
    ctx.storage_manager.submit_prefetch_task.return_value = handle

    keys = [_make_key(0), _make_key(1), _make_key(2)]
    task_id = controller.p2p_lookup_and_lock(keys, _make_layout_desc())

    found = MagicMock()
    found.count_leading_ones.return_value = 2
    ctx.storage_manager.query_prefetch_status.return_value = found
    obj0 = MagicMock(shm_offset=100, shm_byte_length=10)
    obj1 = MagicMock(shm_offset=200, shm_byte_length=20)
    ctx.storage_manager.unsafe_read.return_value = ([keys[0], keys[1]], [obj0, obj1])

    addresses = controller.p2p_query_lookup_results(task_id)
    assert addresses == [
        TransferChannelAddress(offset=100, size=10),
        TransferChannelAddress(offset=200, size=20),
        TransferChannelAddress(offset=-1, size=0),
    ]


def test_query_lookup_results_exactly_once():
    """Re-querying a completed task returns None (the job is consumed)."""
    controller, ctx = _make_controller()
    ctx.storage_manager.submit_prefetch_task.return_value = MagicMock(
        l1_found_indices=()
    )
    task_id = controller.p2p_lookup_and_lock([_make_key(0)], _make_layout_desc())

    found = MagicMock()
    found.count_leading_ones.return_value = 0
    ctx.storage_manager.query_prefetch_status.return_value = found

    assert controller.p2p_query_lookup_results(task_id) == [
        TransferChannelAddress(offset=-1, size=0)
    ]
    assert controller.p2p_query_lookup_results(task_id) is None


def test_query_lookup_results_unknown_task():
    """Querying an unknown task id returns None."""
    controller, _ = _make_controller()
    assert controller.p2p_query_lookup_results(999) is None


def test_query_lookup_results_in_progress():
    """A lookup whose prefetch is not done yet returns None without consuming
    the job."""
    controller, ctx = _make_controller()
    ctx.storage_manager.submit_prefetch_task.return_value = MagicMock(
        l1_found_indices=()
    )
    task_id = controller.p2p_lookup_and_lock([_make_key(0)], _make_layout_desc())

    ctx.storage_manager.query_prefetch_status.return_value = None
    assert controller.p2p_query_lookup_results(task_id) is None
    # Job is still alive; status flips to done on the next poll.
    found = MagicMock()
    found.count_leading_ones.return_value = 0
    ctx.storage_manager.query_prefetch_status.return_value = found
    assert controller.p2p_query_lookup_results(task_id) is not None


def test_unlock_objects_calls_finish_read_prefetched():
    """p2p_unlock_objects forwards the keys to finish_read_prefetched."""
    controller, ctx = _make_controller()
    keys = [_make_key(0), _make_key(1)]
    controller.p2p_unlock_objects(keys)
    ctx.storage_manager.finish_read_prefetched.assert_called_once_with(keys)


def test_unlock_objects_empty_is_noop():
    """Unlocking an empty key list does nothing."""
    controller, ctx = _make_controller()
    controller.p2p_unlock_objects([])
    ctx.storage_manager.finish_read_prefetched.assert_not_called()


def test_report_status_counts_active_jobs():
    """report_status reflects the number of in-flight lookup jobs."""
    controller, ctx = _make_controller()
    ctx.storage_manager.submit_prefetch_task.return_value = MagicMock(
        l1_found_indices=()
    )
    assert controller.report_status() == {"active_p2p_lookup_jobs": 0}
    controller.p2p_lookup_and_lock([_make_key(0)], _make_layout_desc())
    assert controller.report_status() == {"active_p2p_lookup_jobs": 1}


def test_get_handlers_covers_all_p2p_request_types():
    """get_handlers wires exactly the three P2P request types."""
    controller, _ = _make_controller()
    request_types = {spec.request_type for spec in controller.get_handlers()}
    assert request_types == {
        RequestType.P2P_LOOKUP_AND_LOCK,
        RequestType.P2P_QUERY_LOOKUP_RESULTS,
        RequestType.P2P_UNLOCK_OBJECTS,
    }
