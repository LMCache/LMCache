# SPDX-License-Identifier: Apache-2.0
"""Mapped Device-DAX views, visibility ordering, and teardown lifecycle."""

# Standard
from collections.abc import Iterator
from contextlib import closing, nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
import mmap

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.shared_l1.backend import SharedDevDaxL1Backend
from lmcache.v1.distributed.shared_l1.visibility import ACQUIRE, PUBLISH
from lmcache.v1.memory_coordinator.api import WriteGrant, WriteReserveItem
from lmcache.v1.memory_coordinator.pool import MemoryPool
import lmcache.v1.distributed.shared_l1.backend as backend_module

# Local
from .conftest import InProcessCoordinatorClient, RecordingVisibility

_MAPPING_OFFSET = mmap.PAGESIZE


def _key(seed: int = 1) -> ObjectKey:
    return ObjectKey(seed.to_bytes(4, "big"), "model", 0)


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([4, 4])], [torch.float16])


def _backend(
    path: Path,
    pool: MemoryPool,
    **overrides: Any,
) -> SharedDevDaxL1Backend:
    client = overrides.pop("client", InProcessCoordinatorClient(pool))
    visibility = overrides.pop("visibility", RecordingVisibility())
    options: dict[str, Any] = dict(
        devdax_path=str(path),
        capacity_bytes=4096,
        alignment_bytes=64,
        region_id="region",
        layout_id="layout",
        mapping_offset_bytes=_MAPPING_OFFSET,
        coordinator_endpoint="http://127.0.0.1:9400",
        coordinator_token_file="/unused/token",
        visibility_library_path="/unused/visibility.so",
    )
    with (
        patch.object(
            backend_module, "MemoryCoordinatorHttpClient", return_value=client
        ),
        patch.object(
            backend_module, "NativeDeviceDaxVisibility", return_value=visibility
        )
        if visibility is not None
        else nullcontext(),
    ):
        return SharedDevDaxL1Backend(**(options | overrides))


@pytest.fixture
def backend(
    region_file: Path, region_pool: MemoryPool
) -> Iterator[SharedDevDaxL1Backend]:
    with closing(_backend(region_file, region_pool)) as backend:
        yield backend


@pytest.fixture(autouse=True)
def cuda_device(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    device = MagicMock()
    device.pin_memory.return_value = device.unpin_memory.return_value = True
    monkeypatch.setattr(backend_module, "current_device_spec", device)
    return device


def test_two_backends_share_one_physical_tensor_with_exact_ranges(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    producer_visibility = RecordingVisibility()
    consumer_visibility = RecordingVisibility()
    with (
        closing(
            _backend(region_file, region_pool, visibility=producer_visibility)
        ) as producer,
        closing(
            _backend(region_file, region_pool, visibility=consumer_visibility)
        ) as consumer,
    ):
        key = _key()
        write_obj = producer.reserve_write([key], _layout())[0]
        assert write_obj is not None
        expected = torch.arange(write_obj.get_size(), dtype=torch.uint8)
        write_obj.raw_data.copy_(expected)
        producer.finish_write([key])

        read_obj = consumer.reserve_read([key])[0]
        assert read_obj is not None
        assert torch.equal(read_obj.raw_data, expected)
        # One immutable handle owns one reusable view.
        assert consumer.reserve_read([key])[0] is read_obj

        hit = region_pool.lookup([key.to_encoded_object_key()])[0]
        assert hit is not None
        handle = hit.handle
        expected_call = (
            _MAPPING_OFFSET + handle.offset,
            handle.length,
            handle.generation,
        )
        assert producer_visibility.calls == [(PUBLISH, *expected_call)]
        assert consumer_visibility.calls == [
            (ACQUIRE, *expected_call),
            (ACQUIRE, *expected_call),
        ]


def test_view_addresses_are_base_plus_offset_and_bounds_checked(
    backend: SharedDevDaxL1Backend,
) -> None:
    objs = backend.reserve_write([_key(1), _key(2)], _layout())
    base = backend.get_l1_memory_desc().ptr
    for obj in objs:
        assert obj is not None
        assert obj.raw_data.data_ptr() == base + obj.metadata.address
        assert obj.metadata.address + obj.get_size() <= 4096
    backend.abort_write([_key(1), _key(2)])


def test_reordered_grants_are_rejected_before_exposing_views(
    backend: SharedDevDaxL1Backend,
    region_pool: MemoryPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reserve_writes = region_pool.reserve_writes

    def reversed_grants(
        items: list[WriteReserveItem],
    ) -> list[WriteGrant | None]:
        return list(reversed(reserve_writes(items)))

    monkeypatch.setattr(region_pool, "reserve_writes", reversed_grants)
    with pytest.raises(ValueError, match="another key"):
        backend.reserve_write([_key(1), _key(2)], _layout())
    assert region_pool.status().object_count == 0


def test_contract_mismatch_fails_before_mapping(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    with pytest.raises(ValueError, match="contract mismatch"):
        _backend(region_file, region_pool, region_id="wrong-region")


@pytest.mark.parametrize("operation", [PUBLISH, ACQUIRE])
def test_visibility_failure_keeps_object_unpublished_or_releases_read(
    region_file: Path,
    region_pool: MemoryPool,
    operation: int,
) -> None:
    key = _key()
    with closing(
        _backend(
            region_file,
            region_pool,
            visibility=RecordingVisibility(fail_operation=operation),
        )
    ) as backend:
        backend.reserve_write([key], _layout())
        if operation == PUBLISH:
            with pytest.raises(RuntimeError, match="visibility failure"):
                backend.finish_write([key])
            assert region_pool.status().object_count == 0
            return
        backend.finish_write([key])
        with pytest.raises(RuntimeError, match="visibility failure"):
            backend.reserve_read([key])


def test_write_tokens_are_retained_until_finish_or_abort(
    backend: SharedDevDaxL1Backend,
) -> None:
    key = _key()
    backend.reserve_write([key], _layout())
    with pytest.raises(RuntimeError, match="local write reservation"):
        backend.reserve_write([key], _layout())
    backend.finish_write([key])


def test_cuda_registration_failure_is_not_staged(
    region_file: Path,
    region_pool: MemoryPool,
    cuda_device: MagicMock,
) -> None:
    cuda_device.pin_memory.return_value = False
    with pytest.raises(RuntimeError, match="pageable staging is not accepted"):
        _backend(region_file, region_pool)
    cuda_device.unpin_memory.assert_not_called()


def test_cuda_unregistration_failure_keeps_mapping_retryable(
    region_file: Path,
    region_pool: MemoryPool,
    cuda_device: MagicMock,
) -> None:
    cuda_device.unpin_memory.return_value = False
    backend = _backend(region_file, region_pool)
    with pytest.raises(RuntimeError, match="host unregistration failed"):
        backend.close()
    assert backend.memcheck()

    cuda_device.unpin_memory.return_value = True
    backend.close()
    assert not backend.memcheck()


def test_teardown_order_unpins_before_unmap_and_releases_reservations(
    region_file: Path,
    region_pool: MemoryPool,
    cuda_device: MagicMock,
) -> None:
    backend = _backend(region_file, region_pool)
    backend.reserve_write([_key(1)], _layout())
    backend.finish_write([_key(1)])

    backend._mapping = MagicMock(wraps=backend._mapping)
    order = MagicMock()
    order.attach_mock(cuda_device.unpin_memory, "unpin")
    order.attach_mock(backend._mapping.close, "unmap")
    backend.close()
    assert [call[0] for call in order.mock_calls] == ["unpin", "unmap"]


def test_close_refuses_while_exported_views_are_live(
    backend: SharedDevDaxL1Backend,
) -> None:
    key = _key()
    obj = backend.reserve_write([key], _layout())[0]
    assert obj is not None
    backend.finish_write([key])
    obj.ref_count_up()
    with pytest.raises(RuntimeError, match="exported"):
        backend.close()
    assert backend.memcheck()
    obj.ref_count_down()
    backend.close()
    assert not backend.memcheck()


def test_startup_failure_closes_coordinator_client(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    client = InProcessCoordinatorClient(region_pool)
    with pytest.raises(ValueError, match="visibility library"):
        _backend(
            region_file,
            region_pool,
            visibility=None,
            visibility_library_path="/nonexistent/libvisibility.so",
            client=client,
        )
    assert client.closed


def test_read_view_is_constructed_only_after_coordinator_reservation(
    backend: SharedDevDaxL1Backend,
    region_pool: MemoryPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strong lookup precedes view construction and misses expose no view."""
    key = _key()
    backend.reserve_write([key], _layout())
    backend.finish_write([key])
    order = MagicMock()
    for target, method in ((region_pool, "lookup"), (backend, "_memory_object")):
        spy = MagicMock(wraps=getattr(target, method))
        monkeypatch.setattr(target, method, spy)
        order.attach_mock(spy, method)
    assert backend.reserve_read([key])[0] is not None
    assert [call[0] for call in order.mock_calls] == ["lookup", "_memory_object"]
    order.reset_mock()
    assert backend.reserve_read([_key(7)]) == [None]
    assert [call[0] for call in order.mock_calls] == ["lookup"]
