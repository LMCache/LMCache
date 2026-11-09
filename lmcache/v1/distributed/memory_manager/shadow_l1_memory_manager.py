# SPDX-License-Identifier: Apache-2.0
"""Borrowed objects and the memory manager of a shadow L1.

A shadow L1 stages a :class:`BorrowedMemoryObj` placeholder per key as an
owned L1 stages a write buffer, but the placeholder holds no bytes: the
borrowing L2 adapter's load binds it to the lease it holds at the owner and to
a read of the bytes into a device region at retrieve time.
"""

# Standard
from typing import Callable, Protocol
import threading

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import MemoryFormat, MemoryObj, MemoryObjMetadata

Lease = tuple[Callable[[], bool], Callable[[], None]]
"""``(lease_is_valid, on_release)``: a nonblocking check that the lease still
holds, and the idempotent, nonblocking call that ends it. ``on_release`` runs
under the L1 manager's lock on whichever thread frees the object, so it must
never call back into an L1 manager and may only submit the owner's unlock."""


class PendingRead(Protocol):
    """A read a remote borrower started into a local device region."""

    def is_finished(self) -> bool:
        """Return whether the read stopped, successfully or not."""
        ...

    def succeeded(self) -> bool:
        """Return whether the bytes arrived; only meaningful once finished."""
        ...


class BorrowedMemoryObj(MemoryObj):
    """An object whose bytes a peer owns, held under a lease; no local tensor.

    :class:`ShadowL1MemoryManager` creates it as a placeholder: layout
    metadata, no bytes, no lease. A borrower's load :meth:`bind`\\ s it to a
    lease and to a ``read_into`` that reads the bytes into a registered local
    device region at retrieve time. The tensor accessors report no local
    bytes. ``is_valid()`` is False before binding, after :meth:`release`, and
    once the lease ended.
    """

    def __init__(self, metadata: MemoryObjMetadata) -> None:
        super().__init__(metadata)
        self.valid = True
        self._lock = threading.Lock()
        self._lease: Lease | None = None
        self._read_into: Callable[[int, int], PendingRead] | None = None
        self._source = ""

    @property
    def is_bound(self) -> bool:
        """Whether a borrower bound a lease and a read to this object."""
        return self._lease is not None

    @property
    def is_remote(self) -> bool:
        """Whether the object reads its bytes into a device region; True once
        bound."""
        return self._read_into is not None

    def bind(
        self,
        lease: Lease,
        *,
        read_into: Callable[[int, int], PendingRead],
        address: int,
        source: str,
    ) -> None:
        """Bind the placeholder to its lease and bytes; once per object.

        Args:
            lease: The owner's reservation.
            read_into: ``(region_id, offset) -> PendingRead`` starting a read
                of the bytes into a registered device region.
            address: The owner's address of the bytes.
            source: The owner, e.g. its transfer URL. One shadow serves every
                peer, so ``source`` and ``address`` together tell two views of
                the same bytes from views of different bytes.

        Raises:
            RuntimeError: If the object is already bound.
            ValueError: If ``address`` is negative or ``source`` is empty.
        """
        if self._lease is not None:
            raise RuntimeError("BorrowedMemoryObj is already bound")
        if address < 0 or not source:
            raise ValueError("borrowed bytes need the owner's address and source")
        self._read_into = read_into
        self.meta.address = address
        self._source = source
        self._lease = lease

    @property
    def lease(self) -> Lease:
        """Return the object's lease; RuntimeError if it is not bound."""
        if self._lease is None:
            raise RuntimeError("BorrowedMemoryObj is not bound")
        return self._lease

    def replace_lease(self, lease: Lease) -> None:
        """Put the object under a newer lease of the same bytes; the caller
        owns the previous one."""
        self._lease = lease

    def release(self) -> None:
        """Invalidate the object and end its lease; repeated calls are no-ops."""
        if not self.valid:
            return
        self.invalidate()
        if self._lease is not None:
            self._lease[1]()

    def same_bytes(self, other: "BorrowedMemoryObj") -> bool:
        """Return whether ``other`` is a view of the same bytes as this one."""
        return (
            self.is_bound
            and other.is_bound
            and (self._source, self.meta.address) == (other._source, other.meta.address)
        )

    def read_into(self, region_id: int, offset: int) -> PendingRead:
        """Start reading the bytes into a registered local memory region.

        Args:
            region_id: Local region registered with the borrower's transport.
            offset: Byte offset of the destination within that region.

        Returns:
            The in-flight read to poll; it fails instead of reading when the
            lease has already ended.

        Raises:
            RuntimeError: If the object is not bound.
        """
        if self._read_into is None:
            raise RuntimeError("BorrowedMemoryObj is not bound")
        return self._read_into(region_id, offset)

    # -- MemoryObj: layout and lifetime from the metadata --------------------

    def invalidate(self) -> None:
        """Mark the object unusable."""
        self.valid = False

    def is_valid(self) -> bool:
        """Return whether the object is bound, unreleased and its lease holds."""
        return bool(self.valid) and self._lease is not None and self._lease[0]()

    def get_size(self) -> int:
        """Return the logical size in bytes of the borrowed object."""
        return self.meta.phy_size

    def get_physical_size(self) -> int:
        """Return the size in bytes; borrowed objects have no padding here."""
        return self.meta.phy_size

    def get_shape(self) -> torch.Size:
        """Return the shape of the first object group."""
        return self.meta.shape

    def get_shapes(self) -> list[torch.Size]:
        """Return the shape of every object group."""
        assert self.meta.shapes is not None
        return self.meta.shapes

    def get_dtypes(self) -> list[torch.dtype]:
        """Return the dtype of every object group."""
        assert self.meta.dtypes is not None
        return self.meta.dtypes

    def get_memory_format(self) -> MemoryFormat:
        """Return the layout format from the metadata."""
        return self.meta.fmt

    def get_num_tokens(self) -> int:
        """Return the token count along the format's token dimension."""
        return self.meta.shape[self.meta.fmt.token_dim()]

    def pin(self) -> bool:
        """Count a pin; the shadow L1 frees the object, not a pin count."""
        with self._lock:
            self.meta.pin_count += 1
        return True

    def unpin(self) -> bool:
        """Drop a pin."""
        with self._lock:
            self.meta.pin_count = max(0, self.meta.pin_count - 1)
        return True

    def ref_count_up(self) -> None:
        """Count a reference."""
        with self._lock:
            self.meta.ref_count += 1

    def ref_count_down(self) -> None:
        """Drop a reference; the shadow L1 frees the object, not a count."""
        with self._lock:
            self.meta.ref_count = max(0, self.meta.ref_count - 1)

    def get_ref_count(self) -> int:
        """Return the reference count."""
        with self._lock:
            return self.meta.ref_count

    @property
    def metadata(self) -> MemoryObjMetadata:
        """Return the layout metadata."""
        return self.meta

    @property
    def is_pinned(self) -> bool:
        """Return whether the object is pinned."""
        return self.meta.pin_count > 0

    @property
    def can_evict(self) -> bool:
        """Return False: the owner evicts the bytes, the shadow L1 the view."""
        return False

    def parent(self) -> None:
        """Return None: no local allocator backs a borrowed object."""
        return None

    # -- MemoryObj: no local bytes -------------------------------------------

    @property
    def tensor(self) -> None:
        """Return None: the bytes are not in local memory."""
        return None

    @property
    def raw_tensor(self) -> None:
        """Return None: the bytes are not in local memory."""
        return None

    def get_tensor(self, index: int) -> None:
        """Return None: the bytes are not in local memory."""
        return None

    @property
    def byte_array(self) -> bytes:
        """Raise: the bytes are not in local memory; use :meth:`read_into`.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("BorrowedMemoryObj has no local bytes; use read_into")

    @property
    def data_ptr(self) -> int:
        """Raise: the bytes are not in local memory; use :meth:`read_into`.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("BorrowedMemoryObj has no local bytes; use read_into")


class ShadowL1MemoryManager:
    """Memory manager behind :class:`ShadowL1Manager`: hands out
    :class:`BorrowedMemoryObj` placeholders with no capacity check, since the
    bytes are someone else's, and frees an object by ending its lease.
    Satisfies ``L1MemoryManagerProtocol``."""

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Return ``(SUCCESS, placeholders)``, ``count`` unbound objects with
        ``layout_desc``; never out of memory."""
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
                    shapes=list(layout_desc.shapes),
                    dtypes=list(layout_desc.dtypes),
                )
            )
            for _ in range(count)
        ]
        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """End each object's lease; an unbound placeholder just goes away.

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
