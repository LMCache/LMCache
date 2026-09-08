# SPDX-License-Identifier: Apache-2.0

"""Contract tests for generation-scoped logical prefetch leases."""

# Standard
from unittest.mock import Mock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchRequestSpec,
    TrimPolicy,
)
from lmcache.v1.distributed.storage_manager import StorageManager


def _key(name: bytes = b"key") -> ObjectKey:
    return ObjectKey(chunk_hash=name, model_name="contract-test", kv_rank=0)


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([1])], [torch.float16])


def test_generation_is_validated_and_carried_by_spec_and_handle():
    spec = PrefetchRequestSpec(
        keys=[_key()],
        group_layout_descs={0: _layout()},
        policy=TrimPolicy.SPARSE,
        generation=4,
    )
    handle = PrefetchHandle(
        prefetch_request_id=3,
        external_request_id="request",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=spec.generation,
    )

    assert spec.generation == 4
    assert handle.generation == 4
    with pytest.raises(ValueError, match="generation"):
        PrefetchRequestSpec(
            keys=[_key()],
            group_layout_descs={0: _layout()},
            generation=-1,
        )


def test_l1_only_handles_have_independent_idempotent_cleanup():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage.finish_read_prefetched = Mock()

    first = PrefetchHandle(
        prefetch_request_id=-1,
        external_request_id="first",
        l1_found_indices=(0,),
        l1_hit_chunks=1,
        total_requested_keys=1,
        submit_time=0.0,
    )
    second = PrefetchHandle(
        prefetch_request_id=-1,
        external_request_id="second",
        l1_found_indices=(0,),
        l1_hit_chunks=1,
        total_requested_keys=1,
        submit_time=0.0,
    )
    first_key = _key(b"first")
    second_key = _key(b"second")
    storage._remember_prefetch_handle(first, [first_key], 1)
    storage._remember_prefetch_handle(second, [second_key], 1)

    storage._release_prefetch_lease(first)
    storage._release_prefetch_lease(first)

    assert id(second) in storage._prefetch_handle_metadata
    storage._release_prefetch_lease(second)
    assert storage.finish_read_prefetched.call_count == 2
    assert id(first) in storage._released_prefetch_handles
    assert id(second) in storage._released_prefetch_handles


def test_warm_handle_release_does_not_drop_nonexistent_read_locks():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage.finish_read_prefetched = Mock()

    handle = PrefetchHandle(
        prefetch_request_id=-1,
        external_request_id="warm",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
    )
    storage._remember_prefetch_handle(handle, [_key(b"warm")], 0)

    storage.release_prefetch_task(handle)

    assert storage.finish_read_prefetched.call_count == 0


def test_release_uses_generation_guard_and_is_idempotent():
    storage = StorageManager.__new__(StorageManager)
    storage._prefetch_release_lock = threading.Lock()
    storage._released_prefetch_handles = {}
    storage._prefetch_handle_metadata = {}
    storage._prefetch_controller = Mock()
    storage._prefetch_controller.cancel_prefetch_request.return_value = False
    storage.query_prefetch_status = Mock(return_value=None)
    storage.finish_read_prefetched = Mock()

    key = _key(b"release")
    handle = PrefetchHandle(
        prefetch_request_id=8,
        external_request_id="request",
        l1_found_indices=(),
        l1_hit_chunks=0,
        total_requested_keys=1,
        submit_time=0.0,
        generation=12,
    )
    storage._remember_prefetch_handle(handle, [key], 2)

    storage.release_prefetch_task(handle, [key])
    storage.release_prefetch_task(handle, [key])

    storage._prefetch_controller.cancel_prefetch_request.assert_called_once_with(
        8, generation=12
    )
    storage._prefetch_controller.forget_prefetch_result.assert_called_once_with(
        8, generation=12
    )
    storage.finish_read_prefetched.assert_called_once_with([key], read_locks=2)
