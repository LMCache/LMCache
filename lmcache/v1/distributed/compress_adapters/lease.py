# SPDX-License-Identifier: Apache-2.0
"""Exclusive output-range leases for asynchronous GPU decompression.

The existing cache contexts expose fixed staging tensor views but do not own a
standalone slot allocator. This module supplies a compression-local reservation
authority over their physical byte ranges. Production integration must keep one
manager per cache context and route every compressed staging reservation for
that context through it.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
import threading
import weakref

# First Party
from lmcache.v1.distributed.compress_adapters.device import (
    DeviceBufferRange,
    DeviceExecutionContext,
    DeviceIdentity,
)

_MANAGER_CONSTRUCTION_TOKEN = object()


def _ranges_overlap(
    left: DeviceBufferRange,
    right: DeviceBufferRange,
) -> bool:
    """Compare snapshotted physical ranges using half-open byte intervals.

    Args:
        left: First range derived from a contiguous tensor view.
        right: Second range, which may alias a different view of the same
            allocation.

    Returns:
        True if both ranges share a writable byte on the same device. Empty
        and adjacent ranges do not overlap.

    Notes:
        This compares metadata only; callers must keep the backing allocations
        stable while reservations are active.
    """
    if left.device != right.device:
        return False
    if left.byte_length == 0 or right.byte_length == 0:
        return False
    return (
        left.address < right.address + right.byte_length
        and right.address < left.address + left.byte_length
    )


def _require_nonnegative_int(name: str, value: int) -> None:
    """Validate a slot index or length without accepting booleans as integers.

    Args:
        name: Field name used in validation errors.
        value: Integer that must be zero or greater.

    Returns:
        None when the value is valid.

    Raises:
        TypeError: If ``value`` is not an exact ``int``, including ``bool``.
        ValueError: If ``value`` is negative.
    """
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


def _validate_live_range(buffer_range: DeviceBufferRange) -> None:
    """Check that a retained tensor still matches its range snapshot.

    Args:
        buffer_range: Range created by :meth:`DeviceBufferRange.from_tensor`.

    Returns:
        None when contiguity, device, capacity, and base address still match.

    Raises:
        ValueError: If tensor metadata changed or its device no longer matches
            the selected LMCache backend.
        RuntimeError: If Torch cannot expose the tensor's native address.

    Notes:
        This check does not lock tensor storage. The caller must prevent
        external resizing or rebinding throughout the lease lifetime.
    """
    tensor = buffer_range.tensor
    if not tensor.is_contiguous():
        raise ValueError("buffer_range tensor is no longer contiguous")

    current_device = DeviceIdentity.from_device(tensor.device)
    if current_device != buffer_range.device:
        raise ValueError(
            f"buffer_range device changed from {buffer_range.device!r} "
            f"to {current_device!r}"
        )

    current_capacity = tensor.numel() * tensor.element_size()
    if current_capacity != buffer_range.capacity:
        raise ValueError(
            f"buffer_range capacity changed from {buffer_range.capacity} "
            f"to {current_capacity}"
        )

    current_address = tensor.data_ptr()
    if current_address != buffer_range.base_address:
        raise ValueError(
            "buffer_range base address changed from "
            f"{buffer_range.base_address} to {current_address}"
        )


class DeviceOutputLeaseError(RuntimeError):
    """Base error for output-range reservation lifecycle failures."""


class DeviceOutputLeaseUnavailableError(DeviceOutputLeaseError):
    """Raised when a requested range overlaps an active output lease."""


class DeviceOutputLeaseManagerClosedError(DeviceOutputLeaseError):
    """Raised when acquiring from a closed output-lease manager."""


@dataclass(frozen=True, slots=True, init=False, eq=False)
class DeviceOutputLease:
    """Exclusive reservation of one device output byte range.

    Construct leases with :meth:`DeviceOutputLeaseManager.acquire`.

    Attributes:
        buffer_range: Exact staging bytes reserved by the lease.
        batch_idx: Cache-context staging slot containing the range.
        object_group_idx: Object-group view selected within the staging slot.
        owner: Manager that owns the reservation authority.
        lease_id: Manager-local monotonic reservation identifier.

    Notes:
        A lease is an identity-bearing capability rather than a value: two
        leases over the same range are not interchangeable. The manager keeps
        every active lease alive, so dropping the caller's reference cannot
        silently make its range reusable.

        Reservations coordinate callers of the manager; they do not prevent
        external tensor mutation or staging use. Callers must not resize or
        rebind the reserved tensor or its storage while the lease is active.
        Lost active capabilities remain retained until explicit release or
        process exit; closing the cache context does not release them.

        :meth:`release` must be called only after all GPU work that may access
        the range has completed or has been safely drained.
    """

    buffer_range: DeviceBufferRange
    batch_idx: int
    object_group_idx: int
    owner: DeviceOutputLeaseManager = field(repr=False)
    lease_id: int

    @property
    def is_active(self) -> bool:
        """Return whether this lease still reserves its output range."""
        return self.owner.is_active(self)

    def release(self) -> None:
        """Release the reservation idempotently.

        Notes:
            Callers must first ensure that no queued or active GPU operation
            can access :attr:`buffer_range`. A later decompression completion
            owns that sequencing responsibility.
        """
        self.owner.release(self)


class DeviceOutputLeaseManager:
    """Thread-safe reservation authority for one execution context.

    Construct managers with :meth:`for_execution_context`. Repeated calls for
    the same live cache context return the same manager.

    Notes:
        Production integration must create exactly one manager for each cache
        context and use it for all compressed-output staging reservations on
        that context. The manager derives every range from the cache context's
        public object-group staging-buffer API, so callers cannot substitute an
        unrelated tensor. It prevents overlap among its own leases but cannot
        detect code that bypasses the reservation authority.
    """

    def __init__(
        self,
        execution_context: DeviceExecutionContext,
        *,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _MANAGER_CONSTRUCTION_TOKEN:
            raise TypeError(
                "construct output lease managers with "
                "DeviceOutputLeaseManager.for_execution_context()"
            )
        if not isinstance(execution_context, DeviceExecutionContext):
            raise TypeError(
                "execution_context must be a DeviceExecutionContext, got "
                f"{type(execution_context).__name__}"
            )
        self._execution_context = execution_context
        self._lock = threading.Lock()
        self._active_leases: dict[int, DeviceOutputLease] = {}
        self._next_lease_id = 0
        self._closed = False

    @classmethod
    def for_execution_context(
        cls,
        execution_context: DeviceExecutionContext,
    ) -> DeviceOutputLeaseManager:
        """Return the unique live manager for one cache context.

        Args:
            execution_context: Adapter whose cache context owns the staging
                slots that this manager will reserve. The cache context must
                retain object identity equality and hashing.

        Returns:
            The existing live manager for that exact cache context, or a new
            manager when no live authority exists yet.

        Raises:
            TypeError: If ``execution_context`` has the wrong type or its cache
                context cannot be weakly referenced or overrides object
                identity equality or hashing.
            ValueError: If an existing manager was created with a different
                device or stream snapshot for the same cache context.
            DeviceOutputLeaseManagerClosedError: If the cache context's
                manager was already closed.
        """
        if not isinstance(execution_context, DeviceExecutionContext):
            raise TypeError(
                "execution_context must be a DeviceExecutionContext, got "
                f"{type(execution_context).__name__}"
            )
        cache_context = execution_context.cache_context
        context_type = type(cache_context)
        if (
            context_type.__eq__ is not object.__eq__
            or context_type.__hash__ is not object.__hash__
        ):
            raise TypeError(
                "cache context must use object identity equality and hashing"
            )
        try:
            with _MANAGER_REGISTRY_LOCK:
                if cache_context in _CLOSED_MANAGER_CONTEXTS:
                    raise DeviceOutputLeaseManagerClosedError(
                        "output lease manager for this cache context is closed"
                    )
                manager_reference = _MANAGERS_BY_CONTEXT.get(cache_context)
                manager = manager_reference() if manager_reference is not None else None
                if manager is not None:
                    if (
                        manager.execution_context.device != execution_context.device
                        or manager.execution_context.stream
                        is not execution_context.stream
                    ):
                        raise ValueError(
                            "cache context already has an output lease manager "
                            "for a different device or stream snapshot"
                        )
                    return manager

                manager = cls(
                    execution_context,
                    _construction_token=_MANAGER_CONSTRUCTION_TOKEN,
                )
                _MANAGERS_BY_CONTEXT[cache_context] = weakref.ref(manager)
                return manager
        except TypeError as exc:
            raise TypeError(
                "cache context must support weak references and identity hashing"
            ) from exc

    @property
    def execution_context(self) -> DeviceExecutionContext:
        """Return the retained cache-context and stream adapter."""
        return self._execution_context

    @property
    def active_lease_count(self) -> int:
        """Return the number of ranges currently reserved."""
        with self._lock:
            return len(self._active_leases)

    @property
    def is_closed(self) -> bool:
        """Return whether the manager permanently rejects new leases."""
        with self._lock:
            return self._closed

    def acquire(
        self,
        *,
        batch_idx: int,
        object_group_idx: int,
        byte_length: int,
    ) -> DeviceOutputLease:
        """Reserve bytes from one cache-context staging slot without waiting.

        Args:
            batch_idx: Existing cache-context staging slot index.
            object_group_idx: Object-group view within that slot.
            byte_length: Nonnegative number of bytes to reserve.

        Returns:
            A lease containing a range derived from the cache context's public
            object-group staging tensor. The manager retains it until
            :meth:`DeviceOutputLease.release` is called.

        Raises:
            TypeError: If an index or length has the wrong type or the cache
                context's public staging-buffer API does not return a tensor.
            ValueError: If an index or length is invalid, the selected staging
                range is out of bounds, its device differs from the execution
                context, or its live tensor metadata changes.
            DeviceOutputLeaseUnavailableError: If it overlaps an active lease.
            DeviceOutputLeaseManagerClosedError: If this manager is closed.
        """
        _require_nonnegative_int("batch_idx", batch_idx)
        _require_nonnegative_int("object_group_idx", object_group_idx)
        _require_nonnegative_int("byte_length", byte_length)

        with self._lock:
            if self._closed:
                raise DeviceOutputLeaseManagerClosedError(
                    "cannot acquire an output lease from a closed manager"
                )
            tensor = self._execution_context.cache_context.get_temp_object_group_buffer(
                batch_idx,
                object_group_idx,
            )
            buffer_range = DeviceBufferRange.from_tensor(
                tensor,
                byte_offset=0,
                byte_length=byte_length,
            )
            if buffer_range.device != self._execution_context.device:
                raise ValueError(
                    f"buffer_range device {buffer_range.device!r} does not "
                    "match execution context device "
                    f"{self._execution_context.device!r}"
                )
            _validate_live_range(buffer_range)

            for active in self._active_leases.values():
                if _ranges_overlap(buffer_range, active.buffer_range):
                    raise DeviceOutputLeaseUnavailableError(
                        f"buffer_range overlaps active output lease {active.lease_id}"
                    )

            lease_id = self._next_lease_id
            self._next_lease_id += 1
            lease = object.__new__(DeviceOutputLease)
            object.__setattr__(lease, "buffer_range", buffer_range)
            object.__setattr__(lease, "batch_idx", batch_idx)
            object.__setattr__(lease, "object_group_idx", object_group_idx)
            object.__setattr__(lease, "owner", self)
            object.__setattr__(lease, "lease_id", lease_id)
            self._active_leases[lease_id] = lease
            with _MANAGER_RETENTION_LOCK:
                _MANAGERS_WITH_ACTIVE_LEASES.add(self)
            return lease

    def is_active(self, lease: DeviceOutputLease) -> bool:
        """Return whether ``lease`` is an active reservation of this manager.

        Args:
            lease: Lease to query.

        Returns:
            ``True`` only when this exact lease remains registered.

        Raises:
            TypeError: If ``lease`` has the wrong type.
        """
        if not isinstance(lease, DeviceOutputLease):
            raise TypeError(
                f"lease must be a DeviceOutputLease, got {type(lease).__name__}"
            )
        if lease.owner is not self:
            return False
        with self._lock:
            return self._active_leases.get(lease.lease_id) is lease

    def release(self, lease: DeviceOutputLease) -> None:
        """Release one owned lease idempotently.

        Args:
            lease: Lease previously returned by :meth:`acquire`.

        Raises:
            TypeError: If ``lease`` has the wrong type.
            ValueError: If another manager owns ``lease``.
        """
        if not isinstance(lease, DeviceOutputLease):
            raise TypeError(
                f"lease must be a DeviceOutputLease, got {type(lease).__name__}"
            )
        if lease.owner is not self:
            raise ValueError("cannot release a lease owned by another manager")

        with self._lock:
            active = self._active_leases.get(lease.lease_id)
            if active is None:
                return
            if active is not lease:
                raise ValueError("lease identity does not match active reservation")
            del self._active_leases[lease.lease_id]
            if not self._active_leases:
                with _MANAGER_RETENTION_LOCK:
                    _MANAGERS_WITH_ACTIVE_LEASES.discard(self)

    def close(self) -> None:
        """Permanently stop acquisition after all leases are released.

        Raises:
            DeviceOutputLeaseError: If active leases still reserve staging
                ranges. The manager remains open so they can be finalized.

        Notes:
            Repeated calls after a successful close are safe.
        """
        cache_context = self._execution_context.cache_context
        with _MANAGER_REGISTRY_LOCK:
            with self._lock:
                if self._closed:
                    return
                if self._active_leases:
                    raise DeviceOutputLeaseError(
                        "cannot close output lease manager with "
                        f"{len(self._active_leases)} active lease(s)"
                    )
                self._closed = True

            manager_reference = _MANAGERS_BY_CONTEXT.get(cache_context)
            if manager_reference is not None and manager_reference() is self:
                del _MANAGERS_BY_CONTEXT[cache_context]
            _CLOSED_MANAGER_CONTEXTS.add(cache_context)


_MANAGER_REGISTRY_LOCK = threading.Lock()
_MANAGER_RETENTION_LOCK = threading.Lock()
_MANAGERS_BY_CONTEXT: weakref.WeakKeyDictionary[
    object,
    weakref.ReferenceType[DeviceOutputLeaseManager],
] = weakref.WeakKeyDictionary()
_CLOSED_MANAGER_CONTEXTS: weakref.WeakSet[object] = weakref.WeakSet()
_MANAGERS_WITH_ACTIVE_LEASES: set[DeviceOutputLeaseManager] = set()
