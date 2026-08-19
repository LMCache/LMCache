# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Union
import ctypes
import mmap
import os
import stat
import threading

# Third Party
import torch

# First Party
from lmcache.utils import _lmcache_nvtx_annotate, get_size_bytes
from lmcache.v1.memory_allocators.buffer_allocator import BufferAllocator
from lmcache.v1.memory_allocators.mixed_memory_allocator import MixedMemoryAllocator
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.memory_management import (
    AddressManager,
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    TensorMemoryObj,
    logger,
)
import lmcache.v1.memory_management as memory_management

# sysfs exposes every character device under ``/sys/dev/char/<major>:<minor>``
# as a symlink to its device directory. Resolving a device through its number
# rather than its path name keeps aliases (``/dev//dax1.0``, symlinks,
# ``/dev/char/<major>:<minor>``) from bypassing the Device-DAX checks.
_SYSFS_CHAR_ROOT = "/sys/dev/char"
_DAX_BUS_ROOT = "/sys/bus/dax/devices"
_DAX_SUBSYSTEM = "dax"


@dataclass(frozen=True)
class _DeviceIdentity:
    """Identity of an opened backing object, independent of the path used.

    Character devices are identified by their device number: every node or
    symlink with the same ``major:minor`` is the same device. Other files
    (regular files used as test backing) are identified by ``(st_dev,
    st_ino)``.
    """

    is_char_device: bool
    device_number: int
    inode: int

    @classmethod
    def from_stat(cls, fd_status: os.stat_result) -> "_DeviceIdentity":
        if stat.S_ISCHR(fd_status.st_mode):
            return cls(True, fd_status.st_rdev, 0)
        return cls(False, fd_status.st_dev, fd_status.st_ino)


def _sysfs_char_device_dir(st_rdev: int) -> str | None:
    """Return the sysfs device directory of a character device.

    Args:
        st_rdev: The device number reported by ``fstat`` on the opened node.

    Returns:
        The resolved device directory, or ``None`` when sysfs does not expose
        the device (sysfs not mounted, or masked in a container).
    """
    link = os.path.join(_SYSFS_CHAR_ROOT, f"{os.major(st_rdev)}:{os.minor(st_rdev)}")
    try:
        if not os.path.isdir(link):
            return None
        return os.path.realpath(link)
    except OSError:
        return None


def _sysfs_subsystem(device_dir: str) -> str | None:
    """Return the subsystem (bus) name a sysfs device directory belongs to."""
    try:
        return os.path.basename(os.readlink(os.path.join(device_dir, "subsystem")))
    except OSError:
        return None


def _dax_bus_device_dir(major: int, minor: int) -> str | None:
    """Find a device on the dax bus listing by its device number.

    Used when ``/sys/dev/char`` is not visible but ``/sys/bus/dax/devices``
    is. Every entry's ``dev`` attribute holds ``major:minor``.

    Args:
        major: Device major number.
        minor: Device minor number.

    Returns:
        The resolved device directory, or ``None`` when no entry matches.
    """
    wanted = f"{major}:{minor}"
    try:
        names = os.listdir(_DAX_BUS_ROOT)
    except OSError:
        return None
    for name in names:
        entry = os.path.join(_DAX_BUS_ROOT, name)
        try:
            with open(os.path.join(entry, "dev")) as f:
                if f.read().strip() == wanted:
                    return os.path.realpath(entry)
        except OSError:
            continue
    return None


def _resolve_dax_device_dir(device_path: str, st_rdev: int) -> str:
    """Return the sysfs directory of the Device-DAX device behind ``st_rdev``.

    ``/sys/dev/char/<major>:<minor>`` is tried first and must belong to the
    ``dax`` subsystem; when that tree is not visible, or its ``subsystem``
    link cannot be read, the dax bus listing is searched by device number. A
    device that sysfs does not expose cannot be verified and is refused.

    Args:
        device_path: Path the caller used, for error messages.
        st_rdev: Device number of the opened character device.

    Returns:
        The device's sysfs directory.

    Raises:
        ValueError: If the device is on another subsystem or sysfs does not
            expose it.
    """
    major, minor = os.major(st_rdev), os.minor(st_rdev)
    device_dir = _sysfs_char_device_dir(st_rdev)
    if device_dir is not None:
        subsystem = _sysfs_subsystem(device_dir)
        if subsystem == _DAX_SUBSYSTEM:
            return device_dir
        if subsystem is not None:
            raise ValueError(
                f"{device_path} is not a Device-DAX device "
                f"(sysfs subsystem: {subsystem})"
            )
    # No char entry, or its subsystem link is unreadable: the dax bus listing
    # can still identify the device positively.
    bus_dir = _dax_bus_device_dir(major, minor)
    if bus_dir is not None:
        return bus_dir
    raise ValueError(
        f"cannot verify {device_path} as a Device-DAX device: sysfs exposes "
        f"neither a readable {_SYSFS_CHAR_ROOT}/{major}:{minor} subsystem nor "
        f"a {_DAX_BUS_ROOT} entry with that device number (mount sysfs in "
        "the container)"
    )


def _sysfs_attr_int(device_dir: str, attribute: str) -> int | None:
    """Read an integer sysfs attribute.

    Args:
        device_dir: The device's sysfs directory.
        attribute: Attribute file name, e.g. ``size`` or ``align``.

    Returns:
        The attribute value, or ``None`` when it is missing or unreadable.
    """
    try:
        with open(os.path.join(device_dir, attribute)) as f:
            return int(f.read().strip(), 0)
    except (OSError, ValueError):
        return None


class DevDaxNotMappedError(ValueError):
    """No Device-DAX arena is mapped from the requested device path."""


class DevDaxArenaState(Enum):
    """Lifecycle state of one Device-DAX arena in the L1 pool.

    Arenas held in the pool are either ``ACTIVE`` or ``DRAINING``. ``REMOVED`` is
    a terminal, report-only value: once an arena reaches it the arena has been
    unmapped and dropped from the pool, so it is never observed by an in-pool
    lookup.
    """

    ACTIVE = "active"
    DRAINING = "draining"
    REMOVED = "removed"


class DevDaxRemoveMode(Enum):
    """Strategy for retiring a Device-DAX arena at runtime.

    Only ``DRAIN`` is supported today: the arena stops accepting new allocations
    and is unmapped once its already-issued allocations are all freed. Modes that
    relocate live objects (migrate/evict) are intentionally deferred; L1 hands out
    raw device pointers that live requests read and write, so relocation requires
    hooking L1 eviction and is out of scope for the runtime add/remove path.
    """

    DRAIN = "drain"


@dataclass(frozen=True)
class DevDaxArenaStatus:
    """Immutable snapshot of one Device-DAX arena's runtime state.

    Attributes:
        device_path: Canonical path (symlinks resolved) of the device backing
            this arena.
        size_in_bytes: Mapped size of the arena in bytes.
        used_bytes: Bytes currently handed out to live allocations.
        free_bytes: Bytes still available for allocation.
        active_allocations: Number of allocations not yet freed.
        state: Lifecycle state of the arena.
        is_primary: Whether this arena backs ``get_l1_memory_desc`` and so may
            not be removed. Only the initial arena in a pure Device-DAX (no DRAM)
            configuration is primary.
    """

    device_path: str
    size_in_bytes: int
    used_bytes: int
    free_bytes: int
    active_allocations: int
    state: DevDaxArenaState
    is_primary: bool


@dataclass
class _DevDaxArena:
    """Mutable bookkeeping for one mmap-backed Device-DAX arena.

    Each arena owns its own file descriptor, mmap, flat ``torch.uint8`` view, and
    :class:`TensorMemoryAllocator`. Allocations carry the owning
    :class:`DevDaxMemoryAllocator` as their parent; frees are routed back to the
    correct arena by locating the arena whose ``[base_ptr, base_ptr + size)``
    range contains the object's data pointer. ``identity`` records which
    backing device the mmap was opened on, so a second mapping of the same
    device under another path is refused.
    """

    device_path: str
    size: int
    fd: int
    mmap_obj: mmap.mmap
    mmap_buffer: Any
    buffer: torch.Tensor
    allocator: TensorMemoryAllocator
    is_primary: bool
    identity: _DeviceIdentity
    state: DevDaxArenaState = DevDaxArenaState.ACTIVE
    pinned_ptr: int | None = None
    # Captured at map time so ``contains`` stays correct after a deferred
    # close has dropped the buffer references.
    base_ptr: int = 0

    def __post_init__(self) -> None:
        self.base_ptr = self.buffer.data_ptr()

    def contains(self, ptr: int) -> bool:
        """Return whether ``ptr`` falls within this arena's mapped range."""
        return self.base_ptr <= ptr < self.base_ptr + self.size

    def status(self) -> DevDaxArenaStatus:
        """Return an immutable snapshot of this arena's runtime state."""
        address_manager = self.allocator.address_manager
        total = address_manager.get_heap_size()
        free = address_manager.get_free_size()
        return DevDaxArenaStatus(
            device_path=self.device_path,
            size_in_bytes=self.size,
            used_bytes=total - free,
            free_bytes=free,
            active_allocations=self.allocator.num_active_allocations,
            state=self.state,
            is_primary=self.is_primary,
        )


class DevDaxMemoryAllocator(MemoryAllocatorInterface):
    """Allocates L1 objects from DRAM and/or a pool of Device-DAX arenas.

    The mapped bytes of each arena are exposed as a flat CPU ``torch.uint8``
    tensor so the existing L1 state machine and tensor slicing logic can run
    unchanged while the overflow backing storage is one or more configured
    Device-DAX devices. When a local allocator is provided, local DRAM is tried
    first and the Device-DAX arenas are used as overflow.

    The pool starts with a single arena mapped from the configured device and can
    grow or shrink at runtime via :meth:`add_device` and :meth:`remove_device`.
    Removal is drain-based: the arena stops accepting new allocations and is
    unmapped once its already-issued allocations are all freed. In a pure
    Device-DAX configuration (no DRAM) the initial arena is *primary* -- it backs
    :meth:`get_l1_memory_desc` and may not be removed; in a hybrid DRAM +
    Device-DAX configuration every Device-DAX arena is removable overflow.

    Thread safety:
        ``host_mem_lock`` serializes every mutation of the arena pool and of the
        per-arena allocators (Device-DAX allocation, free, add, and remove). The
        DRAM local allocator provides its own synchronization and is used outside
        this lock.
    """

    def __init__(
        self,
        size: int,
        device_path: str,
        *,
        local_allocator: MixedMemoryAllocator | None = None,
        local_size: int = 0,
        shm_name: str | None = None,
        align_bytes: int = AddressManager.ALIGN_BYTES,
    ) -> None:
        if not device_path:
            raise ValueError("device_path must be a non-empty string")
        if size <= 0:
            raise ValueError("size must be > 0")
        if local_size < 0:
            raise ValueError("local_size must be >= 0")
        if local_size and local_allocator is not None:
            raise ValueError("local_size cannot be used with local_allocator")
        if align_bytes <= 0 or align_bytes & (align_bytes - 1) != 0:
            raise ValueError("align_bytes must be a positive power of two")

        # ``size``/``device_path`` describe the initial arena (kept for
        # backwards compatibility); the live pool is ``self._arenas``.
        self.size = size
        self.device_path = device_path
        self.align_bytes = align_bytes
        self.local_allocator = local_allocator
        if local_size:
            self.local_allocator = MixedMemoryAllocator(
                local_size,
                align_bytes=self.align_bytes,
                shm_name=shm_name,
            )
        self.host_mem_lock = threading.Lock()
        self.buffer_allocator = BufferAllocator("cpu")
        self._arenas: list[_DevDaxArena] = []
        self._unregistered = False

        # Primary (non-removable) only in pure Device-DAX mode. The constructor
        # runs single-threaded, so no lock is needed here.
        self._map_and_append_arena(
            os.path.realpath(device_path),
            size,
            is_primary=self.local_allocator is None,
        )

    @property
    def buffer(self) -> torch.Tensor:
        """Return the primary L1 buffer (DRAM if hybrid, else the primary arena)."""
        if self.local_allocator is not None:
            return self.local_allocator.buffer
        if self._arenas:
            return self._arenas[0].buffer
        return torch.empty(0, dtype=torch.uint8)

    @property
    def devdax_buffer(self) -> torch.Tensor:
        """Return the initial Device-DAX arena buffer, or an empty tensor.

        Retained for backwards compatibility with callers that expect a single
        Device-DAX buffer. The pool may hold additional arenas; use
        :meth:`arena_statuses` to inspect all of them.
        """
        if self._arenas:
            return self._arenas[0].buffer
        return torch.empty(0, dtype=torch.uint8)

    @property
    def devdax_allocator(self) -> TensorMemoryAllocator | None:
        """Return the initial Device-DAX arena allocator, or ``None`` if closed."""
        if self._arenas:
            return self._arenas[0].allocator
        return None

    @property
    def address_manager(self) -> AddressManager | None:
        """Return the initial Device-DAX arena address manager, or ``None``."""
        if self._arenas:
            return self._arenas[0].allocator.address_manager
        return None

    def _open_devdax_mapping(
        self,
        device_path: str,
        size: int,
    ) -> tuple[int, mmap.mmap, Any, torch.Tensor, _DeviceIdentity]:
        """Validate and map one Device-DAX arena.

        Character devices are validated against DAX sysfs metadata. Regular
        files are supported as test backings.

        Args:
            device_path: Path to the Device-DAX device.
            size: Number of bytes to map.

        Returns:
            The mapping resources and identity of the opened device.

        Raises:
            ValueError: If the arguments or device metadata are invalid.
            RuntimeError: If the requested size exceeds the device capacity.
            OSError: If the device cannot be opened or mapped read-write.
        """
        if not device_path:
            raise ValueError("device_path must be a non-empty string")
        if size <= 0:
            raise ValueError("size_in_bytes must be > 0")
        fd: int | None = None
        mmap_obj: mmap.mmap | None = None
        try:
            fd = os.open(device_path, os.O_RDWR)
            fd_status = os.fstat(fd)
            identity = _DeviceIdentity.from_stat(fd_status)
            if not identity.is_char_device and not stat.S_ISREG(fd_status.st_mode):
                raise ValueError(
                    f"{device_path} is neither a character device nor a regular file"
                )
            # ``fstat`` reports 0 for character devices: capacity unknown.
            capacity: int | None = fd_status.st_size or None
            if identity.is_char_device:
                device_dir = _resolve_dax_device_dir(device_path, fd_status.st_rdev)
                align = _sysfs_attr_int(device_dir, "align")
                capacity = _sysfs_attr_int(device_dir, "size")
                if not align or not capacity:
                    raise ValueError(
                        f"cannot verify {device_path}: DAX sysfs attributes are "
                        f"unreadable or zero (align={align}, size={capacity})"
                    )
                if size % align != 0:
                    raise ValueError(
                        f"size ({size}) must be a multiple of the device "
                        f"alignment ({align})"
                    )
            if capacity is not None and size > capacity:
                raise RuntimeError(
                    f"l1 devdax size ({size} bytes) exceeds "
                    f"{device_path} capacity ({capacity} bytes)"
                )

            mmap_obj = mmap.mmap(
                fd,
                size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            array_type = ctypes.c_uint8 * size
            mmap_buffer = array_type.from_buffer(mmap_obj)
            buffer = torch.frombuffer(mmap_buffer, dtype=torch.uint8)
            return fd, mmap_obj, mmap_buffer, buffer, identity
        except Exception:
            if mmap_obj is not None:
                mmap_obj.close()
            if fd is not None:
                os.close(fd)
            raise

    def _map_and_append_arena(
        self,
        device_path: str,
        size: int,
        *,
        is_primary: bool,
    ) -> _DevDaxArena:
        """Map one Device-DAX device and append it to the pool as ACTIVE.

        The caller must hold ``host_mem_lock`` (except during ``__init__``, which
        runs single-threaded).

        Args:
            device_path: Path to the Device-DAX device to map.
            size: Number of bytes to map.
            is_primary: Whether the new arena backs ``get_l1_memory_desc`` and so
                may not be removed.

        Returns:
            The newly mapped arena.

        Raises:
            ValueError: If the mapping arguments are invalid, the device is not
                a Device-DAX device, the size violates the device's advertised
                alignment, or the same device is already mapped under another
                path.
            RuntimeError: If the device capacity is smaller than ``size``.
            OSError: If the device cannot be opened or mapped.
        """
        fd, mmap_obj, mmap_buffer, buffer, identity = self._open_devdax_mapping(
            device_path, size
        )
        arena: _DevDaxArena | None = None
        try:
            # Authoritative duplicate check: the opened device, not the path
            # string, decides whether this device is already in the pool.
            for mapped in self._arenas:
                if mapped.identity == identity:
                    raise ValueError(
                        f"Device-DAX arena {device_path} is already mapped "
                        f"as {mapped.device_path}"
                    )
            allocator = TensorMemoryAllocator(buffer, align_bytes=self.align_bytes)
            arena = _DevDaxArena(
                device_path=device_path,
                size=size,
                fd=fd,
                mmap_obj=mmap_obj,
                mmap_buffer=mmap_buffer,
                buffer=buffer,
                allocator=allocator,
                is_primary=is_primary,
                identity=identity,
            )
            self._register_arena_pin(arena)
        except Exception:
            # Release the mapping on any setup failure; all references into
            # the mmap must be dropped before it can close.
            buffer = torch.empty(0, dtype=torch.uint8)
            mmap_buffer = None
            if arena is not None:
                self._close_arena_locked(arena)
            else:
                mmap_obj.close()
                os.close(fd)
            raise

        self._arenas.append(arena)
        return arena

    def _close_arena_locked(self, arena: _DevDaxArena) -> None:
        """Unpin and unmap one arena. The caller must hold ``host_mem_lock``.

        Every reference into the mmap (the allocator buffer, the arena buffer,
        and the ctypes array exported from the mmap) is released before the mmap
        is closed; otherwise CPython refuses to close a buffer with exported
        pointers.
        """
        self._unregister_arena_pin(arena)
        arena.allocator.buffer = torch.empty(0, dtype=torch.uint8)
        arena.buffer = torch.empty(0, dtype=torch.uint8)
        arena.mmap_buffer = None
        arena.mmap_obj.close()
        os.close(arena.fd)

    def _register_arena_pin(self, arena: _DevDaxArena) -> None:
        if not memory_management.current_device_spec.is_pin_supported:
            return
        if not memory_management.current_device_spec.pin_memory(
            arena.buffer.data_ptr(), arena.size
        ):
            logger.warning(
                "pin_memory failed for Device-DAX L1 mapping %s; "
                "falling back to pageable host copies",
                arena.device_path,
            )
            return
        arena.pinned_ptr = arena.buffer.data_ptr()

    def _unregister_arena_pin(self, arena: _DevDaxArena) -> None:
        if arena.pinned_ptr is None:
            return
        memory_management.current_device_spec.unpin_memory(arena.pinned_ptr)
        arena.pinned_ptr = None

    def _active_arenas(self) -> list[_DevDaxArena]:
        """Return arenas accepting new allocations. Caller holds ``host_mem_lock``."""
        return [
            arena for arena in self._arenas if arena.state == DevDaxArenaState.ACTIVE
        ]

    def _find_arena_locked(self, device_path: str) -> _DevDaxArena:
        """Return the arena backed by the device at ``device_path``.

        The node's current identity decides: any alias of a mapped device
        resolves to its arena, and a different device that now sits at a
        mapped path is not found. The canonical path recorded at map time is
        consulted only when ``stat`` reports that the node is missing. A
        spelling the OS cannot interpret (for example one with an embedded NUL
        byte) names nothing and is a lookup miss. The caller holds the lock.

        Raises:
            DevDaxNotMappedError: If no arena is mapped at ``device_path``.
        """
        try:
            path_status = os.stat(device_path)
        except ValueError:
            # ``os.stat`` raises ValueError, not OSError, for a path the OS
            # cannot interpret (embedded NUL byte): nothing can be mapped
            # there, so report a miss rather than let the error escape.
            raise DevDaxNotMappedError(
                f"no Device-DAX arena mapped at {device_path!r}"
            ) from None
        except FileNotFoundError:
            identity = None
        except OSError:
            raise DevDaxNotMappedError(
                f"no Device-DAX arena mapped at {device_path!r}"
            ) from None
        else:
            identity = _DeviceIdentity.from_stat(path_status)
        if identity is not None:
            for arena in self._arenas:
                if arena.identity == identity:
                    return arena
        else:
            canonical = os.path.realpath(device_path)
            for arena in self._arenas:
                if arena.device_path == canonical:
                    return arena
        raise DevDaxNotMappedError(f"no Device-DAX arena mapped at {device_path}")

    def _arena_for_obj_locked(self, memory_obj: MemoryObj) -> _DevDaxArena:
        """Return the arena that owns ``memory_obj``. Caller holds the lock.

        Resolution is by data pointer, so it must run before the object is freed
        (freeing invalidates the object and drops its buffer slice).

        Raises:
            ValueError: If no arena owns the object.
        """
        ptr = memory_obj.data_ptr
        for arena in self._arenas:
            if arena.contains(ptr):
                return arena
        raise ValueError("Memory object does not belong to any Device-DAX arena")

    def _maybe_reap_arena_locked(self, arena: _DevDaxArena) -> None:
        """Unmap a DRAINING arena once it has no live allocations. If lingering
        external views block the unmap, it stays DRAINING and later frees
        retry. Caller holds ``host_mem_lock``."""
        if arena.state != DevDaxArenaState.DRAINING:
            return
        if arena.allocator.num_active_allocations > 0:
            return
        # No live L1 allocations, but a device transfer reading this mapping may
        # still be in flight (L1 hands out raw pinned pointers; some connectors
        # release the pin after only a device-side stream wait). Fence before
        # cudaHostUnregister/munmap so no queued transfer outlives the mapping.
        if memory_management.torch_dev.is_available():
            memory_management.torch_dev.synchronize()
        try:
            self._close_arena_locked(arena)
        except BufferError:
            logger.warning(
                "Device-DAX arena %s has no live allocations but its mapping "
                "is still referenced by exported views; deferring unmap",
                arena.device_path,
            )
            return
        self._arenas.remove(arena)
        arena.state = DevDaxArenaState.REMOVED

    def _reap_draining_arenas_locked(self) -> None:
        """Retry reaping every draining arena. Caller holds ``host_mem_lock``."""
        for arena in list(self._arenas):
            self._maybe_reap_arena_locked(arena)

    def _is_local_obj(self, memory_obj: MemoryObj) -> bool:
        return (
            self.local_allocator is not None
            and memory_obj.parent() is self.local_allocator
        )

    def is_devdax_obj(self, memory_obj: MemoryObj) -> bool:
        """Return whether ``memory_obj`` lives in the Device-DAX arena.

        ``False`` means the object landed in the local DRAM pool of a
        hybrid DRAM+DAX configuration.

        Args:
            memory_obj: An object allocated by this allocator.

        Returns:
            ``True`` for the Device-DAX arena, ``False`` for local DRAM.
        """
        return memory_obj.parent() is self

    def _available_count(
        self,
        address_manager: AddressManager,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> int:
        """Return how many objects of the given shape fit in ``address_manager``."""
        shapes, dtypes = self._adapt_shapes_and_dtypes(shapes, dtypes)
        unit_raw_size = get_size_bytes(shapes, dtypes)
        unit_aligned_size = address_manager.compute_aligned_size(unit_raw_size)
        if unit_aligned_size <= 0:
            return 0
        return address_manager.get_free_size() // unit_aligned_size

    def _local_available_count(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
    ) -> int:
        if self.local_allocator is None:
            return 0
        local_pin_allocator = self.local_allocator.pin_allocator
        assert isinstance(local_pin_allocator, TensorMemoryAllocator)
        return self._available_count(
            local_pin_allocator.address_manager, shapes, dtypes
        )

    def _local_memory_usage(self) -> tuple[int, int]:
        if self.local_allocator is None:
            return 0, 0
        local_pin_allocator = self.local_allocator.pin_allocator
        assert isinstance(local_pin_allocator, TensorMemoryAllocator)
        local_total = local_pin_allocator.address_manager.get_heap_size()
        local_used = local_total - local_pin_allocator.address_manager.get_free_size()
        return local_used, local_total

    def _allocate_from_arenas(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """Allocate one object from the first active arena that can serve it."""
        with self.host_mem_lock:
            for arena in self._active_arenas():
                # Skip full arenas: a failed probe logs a warning per miss,
                # flooding the log once earlier arenas fill up.
                if (
                    self._available_count(
                        arena.allocator.address_manager, shapes, dtypes
                    )
                    <= 0
                ):
                    continue
                obj = arena.allocator.allocate(shapes, dtypes, fmt, str(self))
                if obj is not None:
                    if isinstance(obj, TensorMemoryObj):
                        obj.parent_allocator = self
                    return obj
            return None

    def _batched_allocate_from_arenas(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        count: int,
        fmt: MemoryFormat,
    ) -> Optional[List[MemoryObj]]:
        """Fill ``count`` objects greedily across active arenas, all-or-nothing:
        a partial allocation is rolled back and ``None`` returned."""
        with self.host_mem_lock:
            remaining = count
            allocated: list[tuple[_DevDaxArena, list[MemoryObj]]] = []
            for arena in self._active_arenas():
                if remaining == 0:
                    break
                available = self._available_count(
                    arena.allocator.address_manager, shapes, dtypes
                )
                take = min(remaining, available)
                if take <= 0:
                    continue
                objs = arena.allocator.batched_allocate(
                    shapes, dtypes, take, fmt, str(self)
                )
                if not objs:
                    continue
                for obj in objs:
                    if isinstance(obj, TensorMemoryObj):
                        obj.parent_allocator = self
                allocated.append((arena, objs))
                remaining -= len(objs)

            if remaining > 0:
                # Roll back directly on each arena (the lock is already held).
                for arena, objs in allocated:
                    arena.allocator.batched_free(objs, update_stats=False)
                return None

            result: list[MemoryObj] = []
            for _, objs in allocated:
                result.extend(objs)
            return result

    def _free_devdax_obj(self, memory_obj: MemoryObj) -> None:
        with self.host_mem_lock:
            arena = self._arena_for_obj_locked(memory_obj)
            arena.allocator.free(memory_obj)
            if isinstance(memory_obj, TensorMemoryObj):
                memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)
            self._reap_draining_arenas_locked()

    def _batched_free_devdax_objs(
        self,
        memory_objs: List[MemoryObj],
        update_stats: bool,
    ) -> None:
        with self.host_mem_lock:
            # Resolve each object's owning arena before freeing invalidates it.
            groups: dict[int, tuple[_DevDaxArena, list[MemoryObj]]] = {}
            for memory_obj in memory_objs:
                arena = self._arena_for_obj_locked(memory_obj)
                groups.setdefault(id(arena), (arena, []))[1].append(memory_obj)
            for arena, objs in groups.values():
                arena.allocator.batched_free(objs, update_stats=update_stats)
                for memory_obj in objs:
                    if isinstance(memory_obj, TensorMemoryObj):
                        memory_obj.raw_data = torch.empty(0, dtype=torch.uint8)
            self._reap_draining_arenas_locked()

    @_lmcache_nvtx_annotate
    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.allocate(shapes, dtypes, fmt)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            if self.local_allocator is not None:
                obj = self.local_allocator.allocate(shapes, dtypes, fmt, str(self))
                if obj is not None:
                    return obj
            return self._allocate_from_arenas(shapes, dtypes, fmt)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        allocator_type: Optional[str] = None,
    ) -> Optional[List[MemoryObj]]:
        if fmt == MemoryFormat.BINARY_BUFFER:
            return self.buffer_allocator.batched_allocate(
                shapes, dtypes, batch_size, fmt
            )
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            local_objs: list[MemoryObj] = []
            if self.local_allocator is not None:
                local_count = min(
                    batch_size, self._local_available_count(shapes, dtypes)
                )
                if local_count:
                    local_objs = (
                        self.local_allocator.batched_allocate(
                            shapes, dtypes, local_count, fmt, str(self)
                        )
                        or []
                    )

            remaining = batch_size - len(local_objs)
            if remaining == 0:
                return local_objs

            dax_objs = self._batched_allocate_from_arenas(
                shapes, dtypes, remaining, fmt
            )
            if dax_objs is None:
                if local_objs and self.local_allocator is not None:
                    self.local_allocator.batched_free(local_objs, update_stats=False)
                return None
            return local_objs + dax_objs
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        fmt = memory_obj.meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.free(memory_obj)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            if self._is_local_obj(memory_obj):
                assert self.local_allocator is not None
                self.local_allocator.free(memory_obj)
                return
            if self.is_devdax_obj(memory_obj):
                self._free_devdax_obj(memory_obj)
                return
            raise ValueError("Memory object does not belong to DevDaxMemoryAllocator")
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    @_lmcache_nvtx_annotate
    def batched_free(
        self,
        memory_objs: List[MemoryObj],
        allocator_type: Optional[str] = None,
        update_stats: bool = True,
    ) -> None:
        if not memory_objs:
            return

        fmt = memory_objs[0].meta.fmt
        if fmt == MemoryFormat.BINARY_BUFFER:
            self.buffer_allocator.batched_free(memory_objs)
        elif fmt in [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
            MemoryFormat.EC_TD,
        ]:
            local_objs = [
                memory_obj
                for memory_obj in memory_objs
                if self._is_local_obj(memory_obj)
            ]
            devdax_objs = [
                memory_obj
                for memory_obj in memory_objs
                if self.is_devdax_obj(memory_obj)
            ]
            if len(local_objs) + len(devdax_objs) != len(memory_objs):
                raise ValueError(
                    "One or more memory objects do not belong to DevDaxMemoryAllocator"
                )
            if local_objs:
                assert self.local_allocator is not None
                self.local_allocator.batched_free(local_objs, update_stats=update_stats)
            if devdax_objs:
                self._batched_free_devdax_objs(devdax_objs, update_stats=update_stats)
        else:
            raise ValueError(f"Unsupported memory format: {fmt}")

    def add_device(self, device_path: str, size_in_bytes: int) -> DevDaxArenaStatus:
        """Map an additional Device-DAX device and add it to the pool.

        The new arena is immediately available as overflow; existing allocations
        are untouched.

        Args:
            device_path: Path to a readable and writable Device-DAX device that
                is not already mapped by this allocator under any alias. The
                arena records the canonical path.
            size_in_bytes: Number of bytes to map from the device.

        Returns:
            The status of the newly added arena.

        Raises:
            ValueError: If ``device_path`` is empty, ``size_in_bytes`` is not
                positive, the device is already mapped, the character device
                is not a Device-DAX device, or the mapping violates the
                device's advertised alignment.
            RuntimeError: If the allocator is closed or the device capacity is
                smaller than ``size_in_bytes``.
            OSError: If the device cannot be opened or mapped.
        """
        if not device_path:
            raise ValueError("device_path must be a non-empty string")
        device_path = os.path.realpath(device_path)
        with self.host_mem_lock:
            if self._unregistered:
                raise RuntimeError(
                    "cannot add a device to a closed DevDaxMemoryAllocator"
                )
            # Refuse a duplicate before open + mmap when the node can be
            # inspected; the opened fd is checked again in
            # _map_and_append_arena, which is the authoritative check.
            try:
                identity: _DeviceIdentity | None = _DeviceIdentity.from_stat(
                    os.stat(device_path)
                )
            except OSError:
                identity = None
            for arena in self._arenas:
                if identity is not None and arena.identity == identity:
                    raise ValueError(
                        f"Device-DAX arena {device_path} is already mapped "
                        f"as {arena.device_path}"
                    )
            arena = self._map_and_append_arena(
                device_path, size_in_bytes, is_primary=False
            )
            return arena.status()

    def remove_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
    ) -> DevDaxArenaStatus:
        """Retire a Device-DAX arena from the pool.

        The arena stops accepting new allocations. If it has no live allocations
        it is unmapped immediately; otherwise it is left DRAINING and unmapped
        automatically once its last allocation is freed.

        Args:
            device_path: Path (or any alias) of the arena to remove.
            mode: Removal strategy. Only :attr:`DevDaxRemoveMode.DRAIN` is
                supported.

        Returns:
            The arena status after the request: ``REMOVED`` if it was unmapped
            immediately, otherwise ``DRAINING`` with the live allocation count.

        Raises:
            ValueError: If ``mode`` is unsupported, no arena is mapped at
                ``device_path``, or the arena is the primary (non-removable) one.
        """
        if mode != DevDaxRemoveMode.DRAIN:
            raise ValueError(f"unsupported Device-DAX L1 remove mode: {mode}")
        if not device_path:
            raise DevDaxNotMappedError("no Device-DAX arena mapped at ''")
        with self.host_mem_lock:
            arena = self._find_arena_locked(device_path)
            if arena.is_primary:
                raise ValueError(
                    f"cannot remove primary Device-DAX arena {device_path}"
                )
            arena.state = DevDaxArenaState.DRAINING
            self._maybe_reap_arena_locked(arena)
            return arena.status()

    def arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return a status snapshot of every arena currently in the pool."""
        with self.host_mem_lock:
            return [arena.status() for arena in self._arenas]

    def arena_status(self, device_path: str) -> DevDaxArenaStatus:
        """Return the status of the arena mapped at ``device_path``.

        Any alias of the mapped device (a symlink, another spelling, a hard
        link, or a second node with the same device number) resolves to the
        same arena; a different device at a mapped path does not.

        Args:
            device_path: Path of the arena to look up.

        Returns:
            The arena's current status.

        Raises:
            DevDaxNotMappedError: If no arena is mapped at ``device_path``.
        """
        if not device_path:
            raise DevDaxNotMappedError("no Device-DAX arena mapped at ''")
        with self.host_mem_lock:
            return self._find_arena_locked(device_path).status()

    def memcheck(self) -> bool:
        local_ok = True
        if self.local_allocator is not None:
            local_ok = self.local_allocator.memcheck()
        with self.host_mem_lock:
            dax_ok = all(arena.allocator.memcheck() for arena in self._arenas)
        return local_ok and dax_ok

    def get_memory_usage(self) -> tuple[int, int]:
        local_used, local_total = self._local_memory_usage()
        dax_used = 0
        dax_total = 0
        with self.host_mem_lock:
            for arena in self._arenas:
                address_manager = arena.allocator.address_manager
                arena_total = address_manager.get_heap_size()
                dax_used += arena_total - address_manager.get_free_size()
                if arena.state == DevDaxArenaState.ACTIVE:
                    dax_total += arena_total
        return local_used + dax_used, local_total + dax_total

    def close(self) -> None:
        if self._unregistered:
            return
        if memory_management.torch_dev.is_available():
            memory_management.torch_dev.synchronize()

        with self.host_mem_lock:
            if any(
                arena.allocator.num_active_allocations > 0 for arena in self._arenas
            ):
                raise BufferError(
                    "cannot close DevDaxMemoryAllocator with active allocations"
                )
            for arena in self._arenas:
                self._close_arena_locked(arena)
            self._arenas = []

        if self.local_allocator is not None:
            self.local_allocator.close()
            self.local_allocator = None
        self._unregistered = True

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __str__(self) -> str:
        return "DevDaxMemoryAllocator"
