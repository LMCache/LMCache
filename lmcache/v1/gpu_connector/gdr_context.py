# SPDX-License-Identifier: Apache-2.0
"""GPU landing buffers for objects borrowed from a peer and read by RDMA.

A :class:`~lmcache.v1.memory_management.RemoteBorrowedMemoryObj` has no local
bytes; at retrieve time its bytes are read from the peer straight into GPU
memory, then scattered into the engine's paged KV blocks by the block
transfer kernel. The destination is a :class:`GdrLandingRing`: GPU slots that
the transport has registered, in two banks so the reads into one bank overlap
the kernel draining the other.

Registration goes through a :class:`DeviceRegionRegistrar` that the P2P
controller installs when GPU-direct borrowing is enabled, so this module and
the GPU cache contexts stay transport agnostic. Without a registrar no ring is
created and no GPU memory is spent.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading
import time

# Third Party
import torch

# First Party
from lmcache.v1.platform.stream import CompletionEvent, record_completion_event

_BANKS = 2
# Host-side pause between completion polls; reads take tens of microseconds
# to milliseconds, so a short sleep keeps the GIL free without adding latency.
_POLL_INTERVAL_S = 0.00002


class DeviceRegionRegistrar(ABC):
    """Registers device buffers as read destinations with a transport."""

    @property
    @abstractmethod
    def align_bytes(self) -> int:
        """Alignment every registered buffer and read offset must honor."""

    @abstractmethod
    def register(self, ptr: int, size: int, device_index: int) -> int:
        """Register ``[ptr, ptr + size)`` on device ``device_index``.

        Returns:
            The region id readers pass along with a byte offset.
        """

    @abstractmethod
    def unregister(self, region_id: int) -> None:
        """Release a region returned by :meth:`register`."""


_registrar: DeviceRegionRegistrar | None = None
_registrar_lock = threading.Lock()


def set_device_region_registrar(registrar: DeviceRegionRegistrar | None) -> None:
    """Install (or clear) the registrar that GPU cache contexts use.

    Args:
        registrar: The transport's registrar, or None once it is gone.
    """
    global _registrar
    with _registrar_lock:
        _registrar = registrar


def get_device_region_registrar() -> DeviceRegionRegistrar | None:
    """Return the installed registrar, or None when borrowing is disabled."""
    with _registrar_lock:
        return _registrar


@dataclass(frozen=True)
class LandingSlot:
    """One slot of a landing ring, addressed the way a transport expects."""

    region_id: int
    offset: int
    buffer: torch.Tensor
    """Flat uint8 view of the slot."""


class GdrLandingRing:
    """Registered GPU slots that borrowed objects are read into.

    The ring has two banks of ``slots_per_bank`` slots. A retrieve fills one
    bank while the kernel that scatters the other bank runs, then swaps:
    :meth:`acquire` waits until the kernel that last read a bank finished,
    and :meth:`release` records that a new kernel was queued on it. Like the
    cache context's staging buffer, it serves one transfer at a time: the
    retrieve handlers of one instance run serially on its stream.

    Args:
        device: The GPU holding the ring.
        slot_bytes: Bytes per slot, at least the largest object of any object
            group; rounded up to the registrar's alignment.
        slots_per_bank: Slots per bank, the retrieve batch size.
        registrar: Registers the ring with the transport.
    """

    def __init__(
        self,
        device: torch.device,
        slot_bytes: int,
        slots_per_bank: int,
        registrar: DeviceRegionRegistrar,
    ) -> None:
        align = registrar.align_bytes
        self._slot_bytes = -(-slot_bytes // align) * align
        self._slots_per_bank = slots_per_bank
        size = self._slot_bytes * slots_per_bank * _BANKS
        # Over-allocate by one alignment unit so the registered range can
        # start on an aligned address whatever the allocator returns.
        self._storage = torch.empty(size + align, dtype=torch.uint8, device=device)
        start = (-self._storage.data_ptr()) % align
        self._buffer = self._storage[start : start + size]
        self._registrar = registrar
        self._region_id = registrar.register(
            self._buffer.data_ptr(), size, device.index
        )
        self._bank_events: list[CompletionEvent | None] = [None] * _BANKS
        self._closed = False

    @property
    def slots_per_bank(self) -> int:
        """Slots per bank: the number of chunks one batch can land."""
        return self._slots_per_bank

    @property
    def slot_bytes(self) -> int:
        """Bytes per slot, aligned."""
        return self._slot_bytes

    def slot(self, bank: int, index: int) -> LandingSlot:
        """Return slot ``index`` of ``bank``.

        Args:
            bank: 0 or 1.
            index: Slot within the bank, below :attr:`slots_per_bank`.

        Raises:
            IndexError: If ``bank`` or ``index`` is out of range.
        """
        if not 0 <= bank < _BANKS or not 0 <= index < self._slots_per_bank:
            raise IndexError(f"No landing slot ({bank}, {index})")
        offset = (bank * self._slots_per_bank + index) * self._slot_bytes
        return LandingSlot(
            self._region_id,
            offset,
            self._buffer[offset : offset + self._slot_bytes],
        )

    def acquire(self, bank: int) -> None:
        """Block until the kernel that last read ``bank`` has finished.

        Transport reads are not stream-ordered, so the host must know the
        slots are free before it submits reads into them.

        Args:
            bank: 0 or 1.
        """
        event = self._bank_events[bank]
        while event is not None and not event.is_complete():
            time.sleep(_POLL_INTERVAL_S)
        self._bank_events[bank] = None

    def release(self, bank: int, device: torch.device, stream: object) -> None:
        """Record that work reading ``bank`` was just queued on ``stream``.

        Args:
            bank: 0 or 1.
            device: The ring's device.
            stream: The stream the kernel was queued on.
        """
        self._bank_events[bank] = record_completion_event(device, stream)

    def close(self) -> None:
        """Wait for queued kernels, then unregister the ring; idempotent."""
        if self._closed:
            return
        for bank in range(_BANKS):
            self.acquire(bank)
        self._registrar.unregister(self._region_id)
        self._closed = True


def create_landing_ring(
    device: torch.device, slot_bytes: int, slots_per_bank: int
) -> GdrLandingRing | None:
    """Create a landing ring if a registrar is installed.

    Args:
        device: The GPU holding the ring.
        slot_bytes: Largest object of any object group, in bytes.
        slots_per_bank: Slots per bank, the retrieve batch size.

    Returns:
        The ring, or None when GPU-direct borrowing is disabled.
    """
    registrar = get_device_region_registrar()
    if registrar is None:
        return None
    return GdrLandingRing(device, slot_bytes, slots_per_bank, registrar)
