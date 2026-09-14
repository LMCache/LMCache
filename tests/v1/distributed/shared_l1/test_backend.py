# SPDX-License-Identifier: Apache-2.0
"""Mapped Device-DAX views, visibility ordering, and teardown lifecycle."""

# Standard
from pathlib import Path
from types import SimpleNamespace
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
    visibility: RecordingVisibility,
    *,
    capacity: int = 4096,
    alignment: int = 64,
    register_cuda: bool = False,
) -> SharedDevDaxL1Backend:
    return SharedDevDaxL1Backend(
        devdax_path=str(path),
        capacity_bytes=capacity,
        alignment_bytes=alignment,
        region_id="region",
        layout_id="layout",
        mapping_offset_bytes=_MAPPING_OFFSET,
        client=InProcessCoordinatorClient(pool),
        visibility=visibility,
        register_cuda=register_cuda,
    )


def test_two_backends_share_one_physical_tensor_with_exact_ranges(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    producer_visibility = RecordingVisibility()
    consumer_visibility = RecordingVisibility()
    producer = _backend(region_file, region_pool, producer_visibility)
    consumer = _backend(region_file, region_pool, consumer_visibility)
    try:
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
    finally:
        consumer.close()
        producer.close()


def test_view_addresses_are_base_plus_offset_and_bounds_checked(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    backend = _backend(region_file, region_pool, RecordingVisibility())
    try:
        objs = backend.reserve_write([_key(1), _key(2)], _layout())
        base = backend.get_l1_memory_desc().ptr
        for obj in objs:
            assert obj is not None
            assert obj.raw_data.data_ptr() == base + obj.metadata.address
            assert obj.metadata.address + obj.get_size() <= 4096
        backend.abort_write([_key(1), _key(2)])
    finally:
        backend.close()


def test_reordered_grants_are_rejected_before_exposing_views(
    region_file: Path,
    region_pool: MemoryPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _backend(region_file, region_pool, RecordingVisibility())
    reserve_writes = region_pool.reserve_writes

    def reversed_grants(
        items: list[WriteReserveItem],
    ) -> list[WriteGrant | None]:
        return list(reversed(reserve_writes(items)))

    monkeypatch.setattr(region_pool, "reserve_writes", reversed_grants)
    try:
        with pytest.raises(ValueError, match="another key"):
            backend.reserve_write([_key(1), _key(2)], _layout())
        assert region_pool.status().object_count == 0
    finally:
        backend.close()


def test_contract_mismatch_fails_before_mapping(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    with pytest.raises(ValueError, match="contract mismatch"):
        SharedDevDaxL1Backend(
            devdax_path=str(region_file),
            capacity_bytes=4096,
            alignment_bytes=64,
            region_id="wrong-region",
            layout_id="layout",
            mapping_offset_bytes=_MAPPING_OFFSET,
            client=InProcessCoordinatorClient(region_pool),
            visibility=RecordingVisibility(),
            register_cuda=False,
        )


@pytest.mark.parametrize("operation", [PUBLISH, ACQUIRE])
def test_visibility_failure_keeps_object_unpublished_or_releases_read(
    region_file: Path,
    region_pool: MemoryPool,
    operation: int,
) -> None:
    key = _key()
    backend = _backend(
        region_file,
        region_pool,
        RecordingVisibility(fail_operation=operation),
    )
    try:
        backend.reserve_write([key], _layout())
        if operation == PUBLISH:
            with pytest.raises(RuntimeError, match="visibility failure"):
                backend.finish_write([key])
            assert region_pool.status().object_count == 0
            return
        backend.finish_write([key])
        with pytest.raises(RuntimeError, match="visibility failure"):
            backend.reserve_read([key])
    finally:
        backend.close()


def test_write_tokens_are_retained_until_finish_or_abort(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    backend = _backend(region_file, region_pool, RecordingVisibility())
    try:
        key = _key()
        backend.reserve_write([key], _layout())
        # The token is held locally: a second local reservation for the
        # same key is refused before any coordinator call.
        with pytest.raises(RuntimeError, match="local write reservation"):
            backend.reserve_write([key], _layout())
        backend.finish_write([key])
    finally:
        backend.close()


def test_cuda_registration_failure_is_not_staged(
    region_file: Path,
    region_pool: MemoryPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpinned: list[int] = []
    monkeypatch.setattr(
        backend_module,
        "current_device_spec",
        SimpleNamespace(
            pin_memory=lambda _pointer, _length: False,
            unpin_memory=lambda pointer: unpinned.append(pointer),
        ),
    )
    with pytest.raises(RuntimeError, match="pageable staging is not accepted"):
        _backend(
            region_file,
            region_pool,
            RecordingVisibility(),
            register_cuda=True,
        )
    assert unpinned == []


def test_cuda_unregistration_failure_keeps_mapping_retryable(
    region_file: Path,
    region_pool: MemoryPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unpin_succeeds = False
    monkeypatch.setattr(
        backend_module,
        "current_device_spec",
        SimpleNamespace(
            pin_memory=lambda _pointer, _length: True,
            unpin_memory=lambda _pointer: unpin_succeeds,
        ),
    )
    backend = _backend(
        region_file,
        region_pool,
        RecordingVisibility(),
        register_cuda=True,
    )
    with pytest.raises(RuntimeError, match="host unregistration failed"):
        backend.close()
    assert backend.memcheck()

    unpin_succeeds = True
    backend.close()
    assert not backend.memcheck()


def test_teardown_order_unpins_before_unmap_and_releases_reservations(
    region_file: Path,
    region_pool: MemoryPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []

    def _record_unpin(_pointer: int) -> bool:
        order.append("unpin")
        return True

    monkeypatch.setattr(
        backend_module,
        "current_device_spec",
        SimpleNamespace(
            pin_memory=lambda _pointer, _length: True,
            unpin_memory=_record_unpin,
        ),
    )
    backend = _backend(
        region_file,
        region_pool,
        RecordingVisibility(),
        register_cuda=True,
    )
    backend.reserve_write([_key(1)], _layout())
    backend.finish_write([_key(1)])

    class _MappingProxy:
        """Delegate to the real mapping while recording close order."""

        def __init__(self, real: mmap.mmap) -> None:
            self._real = real

        def close(self) -> None:
            order.append("unmap")
            self._real.close()

        def __getattr__(self, name: str):
            return getattr(self._real, name)

    backend._mapping = _MappingProxy(backend._mapping)  # type: ignore[assignment]
    backend.close()
    assert order == ["unpin", "unmap"]


def test_close_refuses_while_exported_views_are_live(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    backend = _backend(region_file, region_pool, RecordingVisibility())
    key = _key()
    obj = backend.reserve_write([key], _layout())[0]
    assert obj is not None
    backend.finish_write([key])
    # A consumer exports the view (extra reference beyond the backend cache).
    obj.ref_count_up()
    with pytest.raises(RuntimeError, match="exported"):
        backend.close()
    assert backend.memcheck()
    # Dropping the export drains the view; close now proceeds.
    obj.ref_count_down()
    backend.close()
    assert not backend.memcheck()


def test_startup_failure_closes_coordinator_client(
    region_file: Path,
    region_pool: MemoryPool,
) -> None:
    client = InProcessCoordinatorClient(region_pool)
    # A missing visibility library is fatal for the shared-L1 configuration
    # and must not leak the connected client.
    with pytest.raises(ValueError, match="visibility library"):
        SharedDevDaxL1Backend(
            devdax_path=str(region_file),
            capacity_bytes=4096,
            alignment_bytes=64,
            region_id="region",
            layout_id="layout",
            mapping_offset_bytes=_MAPPING_OFFSET,
            visibility_library_path="/nonexistent/libvisibility.so",
            client=client,
            register_cuda=False,
        )
    assert client.closed


def test_read_view_is_constructed_only_after_coordinator_reservation(
    region_file: Path,
    region_pool: MemoryPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A strong lookup precedes view construction and misses expose no view."""
    backend = _backend(region_file, region_pool, RecordingVisibility())
    try:
        key = _key()
        backend.reserve_write([key], _layout())
        backend.finish_write([key])

        order: list[str] = []
        original_lookup = region_pool.lookup

        def recording_lookup(keys):
            order.append("coordinator_lookup")
            return original_lookup(keys)

        original_view = backend._memory_object

        def recording_view(handle, layout):
            order.append("view_construction")
            return original_view(handle, layout)

        monkeypatch.setattr(region_pool, "lookup", recording_lookup)
        monkeypatch.setattr(backend, "_memory_object", recording_view)
        assert backend.reserve_read([key])[0] is not None
        assert order == ["coordinator_lookup", "view_construction"]

        # A miss (no VALID object) produces no view, regardless of what any
        # eventually consistent directory might claim.
        order.clear()
        assert backend.reserve_read([_key(7)]) == [None]
        assert order == ["coordinator_lookup"]
    finally:
        backend.close()
