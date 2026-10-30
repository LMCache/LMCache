# SPDX-License-Identifier: Apache-2.0
"""Public-contract tests for decompression output-range leases."""

# Standard
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import gc
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.compress_adapters.device import DeviceExecutionContext
from lmcache.v1.distributed.compress_adapters.lease import (
    DeviceOutputLease,
    DeviceOutputLeaseError,
    DeviceOutputLeaseManager,
    DeviceOutputLeaseManagerClosedError,
    DeviceOutputLeaseUnavailableError,
)
from lmcache.v1.platform.base.device_spec import DeviceSpec
import lmcache.v1.distributed.compress_adapters.device as device_module


class _TestDeviceSpec(DeviceSpec):
    """Test device selection without requiring an accelerator."""

    def __init__(self, device_type: str, backend_name: str) -> None:
        self._device_type = device_type
        self._backend_name = backend_name

    @property
    def device_type(self) -> str:
        return self._device_type

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def torch_module_name(self) -> str:
        return self._device_type


class _TestCacheContext:
    """Public cache-context surface with aliased staging-slot views."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.stream = object()
        backing = torch.empty(32, dtype=torch.uint8)
        self.buffers = {
            (0, 0): backing[0:8],
            (1, 0): backing[8:16],
            (2, 0): backing[4:12],
            (3, 0): backing[16:16],
        }

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return the configured staging view for one slot and group."""
        try:
            return self.buffers[(batch_idx, object_group_idx)]
        except KeyError as exc:
            raise ValueError("invalid staging slot or object group") from exc


class _FailingCacheContext(_TestCacheContext):
    """Context whose valid staging API raises an internal attribute error."""

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Model a context implementation failure after method resolution."""
        raise AttributeError("staging lookup failed")


class _ValueEqualCacheContext(_TestCacheContext):
    """Distinct staging owners that compare equal by device and stream."""

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, _ValueEqualCacheContext)
            and self.device == other.device
            and self.stream is other.stream
        )

    def __hash__(self) -> int:
        return hash((self.device, self.stream))


class _CustomHashCacheContext(_TestCacheContext):
    """Context with identity equality but a mutable hash value."""

    def __hash__(self) -> int:
        return hash(self.stream)


def _select_spec(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_type: str = "cpu",
    backend_name: str = "cpu",
) -> None:
    """Select a test backend without initializing an accelerator runtime.

    Args:
        monkeypatch: Fixture that restores the selected backend after the test.
        device_type: Torch device spelling accepted by the test backend.
        backend_name: LMCache backend name exposed by derived identities.

    Returns:
        None; subsequent adapter calls use the selected test backend.
    """
    monkeypatch.setattr(
        device_module,
        "current_device_spec",
        _TestDeviceSpec(device_type, backend_name),
    )


def _manager(monkeypatch: pytest.MonkeyPatch) -> DeviceOutputLeaseManager:
    """Create an isolated CPU reservation authority with aliased staging slots.

    Args:
        monkeypatch: Fixture used to select the test CPU backend.

    Returns:
        A manager retaining a fresh context with overlapping, adjacent, and
        empty tensor views. Tests must release every acquired lease.
    """
    _select_spec(monkeypatch)
    context = _TestCacheContext(torch.device("cpu"))
    execution_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )
    return DeviceOutputLeaseManager.for_execution_context(execution_context)


def _acquire(
    manager: DeviceOutputLeaseManager,
    *,
    batch_idx: int = 0,
    object_group_idx: int = 0,
    byte_length: int = 8,
) -> DeviceOutputLease:
    """Reserve a test staging range, defaulting to the first eight-byte slot.

    Args:
        manager: Reservation authority created for the test context.
        batch_idx: Staging slot to select.
        object_group_idx: Object-group view within that slot.
        byte_length: Number of bytes to reserve from the view's beginning.

    Returns:
        An active lease that the test must release.

    Raises:
        TypeError: If a slot index or length is not an exact int.
        ValueError: If the selected slot or byte range is invalid.
        DeviceOutputLeaseUnavailableError: If the range is already reserved.
        DeviceOutputLeaseManagerClosedError: If the manager is closed.
    """
    return manager.acquire(
        batch_idx=batch_idx,
        object_group_idx=object_group_idx,
        byte_length=byte_length,
    )


def test_manager_factory_requires_execution_context() -> None:
    """A reservation authority is always bound to a validated context."""
    with pytest.raises(TypeError, match="DeviceExecutionContext"):
        DeviceOutputLeaseManager.for_execution_context(
            object()  # type: ignore[arg-type]
        )


def test_direct_manager_construction_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers cannot create an independent authority for one context."""
    manager = _manager(monkeypatch)

    with pytest.raises(TypeError, match="for_execution_context"):
        DeviceOutputLeaseManager(manager.execution_context)


def test_manager_factory_reuses_context_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct adapters for one cache context resolve to one manager."""
    _select_spec(monkeypatch)
    context = _TestCacheContext(torch.device("cpu"))
    first_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )
    second_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )

    first = DeviceOutputLeaseManager.for_execution_context(first_context)
    second = DeviceOutputLeaseManager.for_execution_context(second_context)

    assert first is second


def test_manager_factory_rejects_value_equal_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal staging owners cannot accidentally share a reservation authority."""
    _select_spec(monkeypatch)
    first = _ValueEqualCacheContext(torch.device("cpu"))
    second = _ValueEqualCacheContext(torch.device("cpu"))
    second.stream = first.stream
    assert first is not second
    assert first == second
    assert hash(first) == hash(second)
    assert first.buffers[(0, 0)].data_ptr() != second.buffers[(0, 0)].data_ptr()

    for context in (first, second):
        execution_context = DeviceExecutionContext.from_cache_context(
            context  # type: ignore[arg-type]
        )
        with pytest.raises(TypeError, match="object identity equality and hashing"):
            DeviceOutputLeaseManager.for_execution_context(execution_context)


def test_manager_factory_rejects_custom_context_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mutable hashing cannot hide an existing manager or closed-state entry."""
    _select_spec(monkeypatch)
    context = _CustomHashCacheContext(torch.device("cpu"))
    execution_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError, match="object identity equality and hashing"):
        DeviceOutputLeaseManager.for_execution_context(execution_context)


def test_manager_factory_is_thread_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent resolution cannot create two authorities for one context."""
    _select_spec(monkeypatch)
    context = _TestCacheContext(torch.device("cpu"))
    execution_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )
    worker_count = 8
    barrier = Barrier(worker_count)

    def resolve_once() -> DeviceOutputLeaseManager:
        barrier.wait()
        return DeviceOutputLeaseManager.for_execution_context(execution_context)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        managers = list(executor.map(lambda _: resolve_once(), range(worker_count)))

    assert all(manager is managers[0] for manager in managers)


def test_active_manager_survives_dropped_caller_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cycle collection cannot silently revoke an active reservation."""
    _select_spec(monkeypatch)
    context = _TestCacheContext(torch.device("cpu"))
    execution_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )
    manager = DeviceOutputLeaseManager.for_execution_context(execution_context)
    manager_reference = weakref.ref(manager)
    _acquire(manager)

    del manager
    gc.collect()

    retained_manager = manager_reference()
    assert retained_manager is not None
    assert retained_manager.active_lease_count == 1
    assert (
        DeviceOutputLeaseManager.for_execution_context(execution_context)
        is retained_manager
    )
    with pytest.raises(DeviceOutputLeaseUnavailableError, match="overlaps"):
        _acquire(retained_manager)


def test_manager_is_collectible_after_last_lease_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The active-manager retention root is removed after final release."""
    manager = _manager(monkeypatch)
    manager_reference = weakref.ref(manager)
    lease = _acquire(manager)

    lease.release()
    del lease
    del manager
    gc.collect()

    assert manager_reference() is None


def test_acquire_derives_range_from_context_staging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Callers select a context slot instead of supplying an arbitrary tensor."""
    manager = _manager(monkeypatch)
    lease = _acquire(manager, batch_idx=0, byte_length=4)
    context = manager.execution_context.cache_context

    assert lease.buffer_range.tensor is context.get_temp_object_group_buffer(0, 0)
    assert lease.buffer_range.byte_offset == 0
    assert lease.buffer_range.byte_length == 4
    assert lease.batch_idx == 0
    assert lease.object_group_idx == 0
    assert lease.owner is manager
    assert lease.lease_id == 0
    assert lease.is_active
    assert manager.active_lease_count == 1

    lease.release()


def test_overlapping_alias_view_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Physical addresses detect overlap across distinct context slot views."""
    manager = _manager(monkeypatch)
    first = _acquire(manager, batch_idx=0)

    with pytest.raises(DeviceOutputLeaseUnavailableError, match="overlaps"):
        _acquire(manager, batch_idx=2)

    first.release()


def test_adjacent_ranges_can_be_leased_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Half-open staging slots that only share a boundary do not overlap."""
    manager = _manager(monkeypatch)
    first = _acquire(manager, batch_idx=0)
    second = _acquire(manager, batch_idx=1)

    assert first.is_active
    assert second.is_active
    assert manager.active_lease_count == 2

    first.release()
    second.release()


def test_empty_ranges_do_not_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty records reserve no writable byte and may share one address."""
    manager = _manager(monkeypatch)
    first = _acquire(manager, batch_idx=3, byte_length=0)
    second = _acquire(manager, batch_idx=3, byte_length=0)

    assert manager.active_lease_count == 2

    first.release()
    second.release()


def test_release_is_idempotent_and_allows_reacquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalization may safely retry release without losing a newer lease."""
    manager = _manager(monkeypatch)
    first = _acquire(manager)

    first.release()
    first.release()
    second = _acquire(manager)

    assert not first.is_active
    assert second.is_active
    assert second.lease_id != first.lease_id
    second.release()


def test_foreign_manager_cannot_release_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the manager that issued a capability may release it."""
    owner = _manager(monkeypatch)
    other = _manager(monkeypatch)
    lease = _acquire(owner)

    assert not other.is_active(lease)
    with pytest.raises(ValueError, match="another manager"):
        other.release(lease)
    assert lease.is_active

    lease.release()


@pytest.mark.parametrize(
    ("arguments", "exception", "message"),
    [
        ((True, 0, 0), TypeError, "batch_idx"),
        ((0, True, 0), TypeError, "object_group_idx"),
        ((0, 0, True), TypeError, "byte_length"),
        ((-1, 0, 0), ValueError, "batch_idx"),
        ((0, -1, 0), ValueError, "object_group_idx"),
        ((0, 0, -1), ValueError, "byte_length"),
        ((0, 0, 9), ValueError, "exceeds tensor capacity"),
    ],
)
def test_acquire_validates_slot_and_range_values(
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[int, int, int],
    exception: type[Exception],
    message: str,
) -> None:
    """Slot selectors and byte bounds retain exact public error semantics."""
    manager = _manager(monkeypatch)

    with pytest.raises(exception, match=message):
        manager.acquire(
            batch_idx=arguments[0],
            object_group_idx=arguments[1],
            byte_length=arguments[2],
        )


def test_acquire_preserves_error_from_staging_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An internal context failure is not mislabeled as a missing method."""
    _select_spec(monkeypatch)
    context = _FailingCacheContext(torch.device("cpu"))
    execution_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )
    manager = DeviceOutputLeaseManager.for_execution_context(execution_context)

    with pytest.raises(AttributeError, match="staging lookup failed"):
        _acquire(manager)


def test_acquire_rejects_staging_tensor_on_wrong_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A context cannot return staging owned by another selected backend."""
    _select_spec(monkeypatch, device_type="cuda", backend_name="cuda")
    context = _TestCacheContext(torch.device("cuda:0"))
    execution_context = DeviceExecutionContext.from_cache_context(
        context  # type: ignore[arg-type]
    )
    manager = DeviceOutputLeaseManager.for_execution_context(execution_context)

    with pytest.raises(ValueError, match="not 'cpu'"):
        _acquire(manager)


def test_close_requires_no_active_leases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown cannot discard a reservation while GPU work may own it."""
    manager = _manager(monkeypatch)
    lease = _acquire(manager)

    with pytest.raises(DeviceOutputLeaseError, match="1 active lease"):
        manager.close()
    assert not manager.is_closed

    lease.release()
    manager.close()
    manager.close()
    assert manager.is_closed

    with pytest.raises(DeviceOutputLeaseManagerClosedError, match="closed"):
        _acquire(manager)
    with pytest.raises(DeviceOutputLeaseManagerClosedError, match="closed"):
        DeviceOutputLeaseManager.for_execution_context(manager.execution_context)


def test_concurrent_overlap_allows_exactly_one_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent callers cannot both reserve the same context staging slot."""
    manager = _manager(monkeypatch)
    worker_count = 8
    barrier = Barrier(worker_count)

    def acquire_once() -> DeviceOutputLease | None:
        barrier.wait()
        try:
            return _acquire(manager)
        except DeviceOutputLeaseUnavailableError:
            return None

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda _: acquire_once(), range(worker_count)))

    leases = [result for result in results if result is not None]
    assert len(leases) == 1
    assert manager.active_lease_count == 1

    leases[0].release()
