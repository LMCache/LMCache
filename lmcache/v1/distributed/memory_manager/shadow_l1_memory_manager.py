# SPDX-License-Identifier: Apache-2.0
"""Borrowed objects and the memory manager of a shadow L1.

A shadow L1 stages a :class:`BorrowedMemoryObj` placeholder per key exactly
as an owned L1 stages a write buffer, but the placeholder holds no bytes. The
lending L2 adapter's load binds it to the lease it holds at the owner and to
the bytes: a local tensor view of them, or a ``read_into`` that moves them
into a registered device region at retrieve time.
"""

# Standard
from typing import Callable, Protocol

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

Lease = tuple[Callable[[], bool], Callable[[], None]]
"""``(lease_is_valid, on_release)``: a nonblocking check that the lease still
holds, and the call that ends it. ``on_release`` is idempotent, nonblocking
and takes no lock the caller might hold: it runs under the L1 manager's lock
on whichever thread frees the object, so it must never call back into an L1
manager and may only submit the owner's unlock, never wait for it."""


class PendingRead(Protocol):
    """A read a remote lender started into a local device region."""

    def is_finished(self) -> bool:
        """Return whether the read stopped, successfully or not."""
        ...

    def succeeded(self) -> bool:
        """Return whether the bytes arrived; only meaningful once finished."""
        ...


class BorrowedMemoryObj(TensorMemoryObj):
    """An object whose bytes someone else owns, held under a lease.

    Created by :class:`ShadowL1MemoryManager` as a placeholder: layout
    metadata, no bytes, no lease. A lender's load :meth:`bind`\\ s it to the
    lease it holds and to the bytes, either a tensor view (``raw_tensor``
    then serves the ordinary copy path) or a ``read_into`` for bytes that are
    read straight into a device region at retrieve time. ``is_valid()`` is
    False before binding, after :meth:`release`, and once the lease ended.

    Args:
        metadata: Layout of the object; ``address`` is set by a remote
            binding.
    """

    def __init__(self, metadata: MemoryObjMetadata) -> None:
        super().__init__(
            torch.empty(0, dtype=torch.uint8), metadata, parent_allocator=None
        )
        self._lease: Lease | None = None
        self._read_into: Callable[[int, int], PendingRead] | None = None

    @property
    def is_bound(self) -> bool:
        """Whether a lender bound bytes and a lease to this object."""
        return self._lease is not None

    @property
    def is_remote(self) -> bool:
        """Whether the bytes are read into a device region at retrieve time."""
        return self._read_into is not None

    def bind(
        self,
        lease: Lease,
        *,
        tensor: torch.Tensor | None = None,
        read_into: Callable[[int, int], PendingRead] | None = None,
        address: int = -1,
    ) -> None:
        """Bind the placeholder to its bytes and lease; once per object.

        Args:
            lease: The owner's reservation as a ``(lease_is_valid,
                on_release)`` pair.
            tensor: A flat view of the borrowed bytes, at least ``get_size()``
                bytes long, for bytes mapped into this process.
            read_into: ``(region_id, offset) -> PendingRead`` starting a read
                of the bytes into a registered device region, for bytes that
                stay remote.
            address: The owner's address of remote bytes, so that two views
                of the same bytes can be told apart from two copies.

        Raises:
            RuntimeError: If the object is already bound.
            ValueError: If not exactly one of ``tensor`` and ``read_into`` is
                given, or ``tensor`` is shorter than the object.
        """
        if self._lease is not None:
            raise RuntimeError("BorrowedMemoryObj is already bound")
        if (tensor is None) == (read_into is None):
            raise ValueError("bind takes exactly one of tensor and read_into")
        if tensor is not None:
            if tensor.numel() * tensor.element_size() < self.get_size():
                raise ValueError(
                    f"borrowed tensor holds {tensor.numel() * tensor.element_size()} "
                    f"bytes, the object needs {self.get_size()}"
                )
            self.raw_data = tensor.view(torch.uint8).flatten()
        else:
            self._read_into = read_into
            self.meta.address = address
        self._lease = lease

    @property
    def lease(self) -> Lease:
        """Return the object's lease.

        Raises:
            RuntimeError: If the object is not bound.
        """
        if self._lease is None:
            raise RuntimeError("BorrowedMemoryObj is not bound")
        return self._lease

    def replace_lease(self, lease: Lease) -> None:
        """Put the object under another lease of the same bytes; the caller
        owns the previous one.

        Args:
            lease: The newer reservation.
        """
        self._lease = lease

    def release(self) -> None:
        """Invalidate the object and end its lease; repeated calls are no-ops."""
        if not self.valid:
            return
        self.invalidate()
        if self._lease is not None:
            self._lease[1]()

    def is_valid(self) -> bool:
        """Return whether the object is bound, unreleased and its lease holds."""
        return bool(self.valid) and self._lease is not None and self._lease[0]()

    def same_bytes(self, other: "BorrowedMemoryObj") -> bool:
        """Return whether ``other`` is a view of the same bytes as this one,
        so that a shadow L1 may renew this object from ``other``'s lease."""
        if not (self.is_bound and other.is_bound) or self.is_remote != other.is_remote:
            return False
        if self.is_remote:
            return self.meta.address == other.meta.address
        return (
            self.raw_data.data_ptr() == other.raw_data.data_ptr()
            and self.get_size() == other.get_size()
        )

    def local_device(self) -> torch.device | None:
        """Return the device a reader touches the bytes on, or None when the
        bytes are read wherever the reader lands them."""
        return None if self.is_remote else self.raw_data.device

    @property
    def tensor(self) -> torch.Tensor | None:
        """The bytes as a tensor; None for remote bytes, which have none here."""
        return None if self.is_remote else super().tensor

    @property
    def raw_tensor(self) -> torch.Tensor | None:
        """The flat borrowed bytes; None for remote bytes."""
        return None if self.is_remote else super().raw_tensor

    @property
    def byte_array(self) -> memoryview:
        """The bytes; remote bytes cannot be read here.

        Raises:
            RuntimeError: If the bytes are remote.
        """
        if self.is_remote:
            raise RuntimeError("BorrowedMemoryObj holds remote bytes; use read_into")
        return super().byte_array

    @property
    def data_ptr(self) -> int:
        """Host address of the bytes; remote bytes have none.

        Raises:
            RuntimeError: If the bytes are remote.
        """
        if self.is_remote:
            raise RuntimeError("BorrowedMemoryObj holds remote bytes; use read_into")
        return super().data_ptr

    def read_into(self, region_id: int, offset: int) -> PendingRead:
        """Start reading remote bytes into a registered local memory region.

        Args:
            region_id: Local region registered with the lender's transport
                (e.g. the GPU staging buffer).
            offset: Byte offset of the destination within that region.

        Returns:
            The in-flight read; poll it until finished. It fails instead of
            reading when the lease has already ended.

        Raises:
            RuntimeError: If the bytes are local; read ``raw_tensor`` instead.
        """
        if self._read_into is None:
            raise RuntimeError("BorrowedMemoryObj holds local bytes; copy raw_tensor")
        return self._read_into(region_id, offset)


class ShadowL1MemoryManager:
    """Memory manager behind :class:`ShadowL1Manager`.

    It hands out :class:`BorrowedMemoryObj` placeholders without checking any
    capacity, since the bytes are someone else's, and frees an object by
    ending its lease. Satisfies ``L1MemoryManagerProtocol``.
    """

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Create ``count`` unbound placeholders with ``layout_desc``.

        Args:
            layout_desc: Layout of the objects.
            count: Number of placeholders.

        Returns:
            ``(SUCCESS, placeholders)``; never out of memory.
        """
        size = sum(
            shape.numel() * dtype.itemsize
            for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=True)
        )
        objects: list[MemoryObj] = [
            BorrowedMemoryObj(
                MemoryObjMetadata(
                    shape=layout_desc.shapes[0],
                    dtype=layout_desc.dtypes[0],
                    address=-1,
                    phy_size=size,
                    ref_count=0,
                    fmt=MemoryFormat.KV_2LTD,
                    shapes=list(layout_desc.shapes),
                    dtypes=list(layout_desc.dtypes),
                )
            )
            for _ in range(count)
        ]
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """End each object's lease; an unbound placeholder just goes away.

        Args:
            mem_objs: Borrowed objects to release.

        Returns:
            ``SUCCESS``.

        Raises:
            TypeError: If an object is not a :class:`BorrowedMemoryObj`,
                which would mean a wiring bug placed local memory here.
        """
        for obj in mem_objs:
            if not isinstance(obj, BorrowedMemoryObj):
                raise TypeError(f"Shadow L1 holds only borrowed objects, got {obj!r}")
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
