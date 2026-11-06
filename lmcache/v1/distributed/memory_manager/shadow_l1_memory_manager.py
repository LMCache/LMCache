# SPDX-License-Identifier: Apache-2.0
"""The borrowed objects a lender hands over, and the memory manager that frees
them."""

# Standard
from typing import Callable
import abc

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import MemoryObj, MemoryObjMetadata, TensorMemoryObj


class BorrowedMemoryObj(MemoryObj):
    """A memory object held under a lease that someone else grants.

    The lender (e.g. an L2 adapter holding a peer's read reservation) decides
    how long the bytes stay valid: ``is_valid()`` is False once the lease
    ended, and :meth:`release` ends it from this side. A lender must release
    an object that is garbage-collected without :meth:`release`. Subclasses
    set ``_lease_is_valid`` and ``_on_release`` in their constructor.
    """

    _lease_is_valid: Callable[[], bool]
    _on_release: Callable[[], None]

    @property
    def lease(self) -> tuple[Callable[[], bool], Callable[[], None]]:
        """Return the object's ``(lease_is_valid, on_release)`` pair."""
        return self._lease_is_valid, self._on_release

    def replace_lease(
        self, lease_is_valid: Callable[[], bool], on_release: Callable[[], None]
    ) -> None:
        """Put the object under another lease, a newer reservation of the
        same bytes. The caller owns the previous lease.

        Args:
            lease_is_valid: Nonblocking check that the new lease still holds.
            on_release: Ends the new lease; same rules as the constructor's.
        """
        self._lease_is_valid = lease_is_valid
        self._on_release = on_release

    def release(self) -> None:
        """Invalidate the object and end the lease; repeated calls are no-ops."""
        self.invalidate()
        self._on_release()

    @abc.abstractmethod
    def same_bytes(self, other: "BorrowedMemoryObj") -> bool:
        """Return whether ``other`` is a view of the same bytes as this one,
        so that a shadow L1 may renew this object from ``other``'s lease."""
        raise NotImplementedError

    @abc.abstractmethod
    def local_device(self) -> torch.device | None:
        """Return the device a reader touches the bytes on, or None when the
        bytes are read wherever the reader lands them."""
        raise NotImplementedError


class BorrowedTensorMemoryObj(TensorMemoryObj, BorrowedMemoryObj):
    """A tensor view of bytes another party owns, held under a lease.

    It allocates nothing locally. Whoever lends the bytes (e.g. an L2 adapter
    holding a peer's read reservation) supplies the lease: ``is_valid()``
    also reports whether the lease still holds, and ``release`` ends it.

    Args:
        raw_data: The borrowed bytes; a view, never a copy.
        metadata: Layout of the object.
        lease_is_valid: Nonblocking check that the lease still holds.
        on_release: Ends the lease; idempotent, nonblocking, and takes no lock
            the caller might hold. It runs under the L1 manager's lock, on
            whichever thread frees the view, so it must never call back into
            an L1 manager and may only submit the peer unlock, never wait for
            it.
    """

    def __init__(
        self,
        raw_data: torch.Tensor,
        metadata: MemoryObjMetadata,
        *,
        lease_is_valid: Callable[[], bool],
        on_release: Callable[[], None],
    ) -> None:
        super().__init__(raw_data, metadata, parent_allocator=None)
        self._lease_is_valid = lease_is_valid
        self._on_release = on_release

    def same_bytes(self, other: BorrowedMemoryObj) -> bool:
        """Both views alias one buffer of one size."""
        return (
            isinstance(other, BorrowedTensorMemoryObj)
            and self.raw_data.data_ptr() == other.raw_data.data_ptr()
            and self.get_size() == other.get_size()
        )

    def local_device(self) -> torch.device:
        """The device the aliased buffer lives on."""
        return self.raw_data.device

    def is_valid(self) -> bool:
        """Return whether the view is unreleased and its lease still holds."""
        return super().is_valid() and self._lease_is_valid()


class ShadowL1MemoryManager:
    """Memory manager behind :class:`ShadowL1Manager`.

    Its objects are :class:`BorrowedMemoryObj` views whose bytes someone else
    owns, so it never allocates and has no capacity; freeing an object ends
    its lease. Satisfies ``L1MemoryManagerProtocol``.
    """

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Refuse every allocation; views enter through ``ShadowL1Manager.admit``.

        Args:
            layout_desc: Ignored.
            count: Ignored.

        Returns:
            ``(OUT_OF_MEMORY, [])``, always.
        """
        return L1Error.OUT_OF_MEMORY, []

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """End each view's lease.

        Args:
            mem_objs: Borrowed views to release.

        Returns:
            ``SUCCESS``.

        Raises:
            TypeError: If an object is not a :class:`BorrowedMemoryObj`,
                which would mean a wiring bug placed local memory here.
        """
        for obj in mem_objs:
            if not isinstance(obj, BorrowedMemoryObj):
                raise TypeError(f"Shadow L1 holds only borrowed views, got {obj!r}")
            obj.release()
        return L1Error.SUCCESS

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(0, 0)``: borrowed bytes are not local memory."""
        return 0, 0

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return None; there is no local buffer to register."""
        return None

    def close(self) -> None:
        """Do nothing; there is nothing to unmap."""

    def memcheck(self) -> bool:
        """Return True; there is no allocator bookkeeping to check."""
        return True
