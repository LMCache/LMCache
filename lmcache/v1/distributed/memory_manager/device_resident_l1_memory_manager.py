# SPDX-License-Identifier: Apache-2.0
"""Device-resident L1 memory manager.

Provides a device-resident L1 tier where L1 entries hold GPU/device memory
objects directly (instead of CPU pinned-DRAM). Retrieve serves via D2D
copy rather than H2D, which is useful when the L2 adapter can DMA directly
into device memory (e.g. NVMe-to-GPU direct read).

The manager is **backend-agnostic**: it interacts with device memory pools
through the :class:`DeviceMemoryPool` protocol. Concrete backends implement
this protocol and are instantiated in ``_init_<backend>_pools`` methods.

This PR provides the framework only — no backend is implemented yet. A
follow-up PR will add the PHX backend (``_init_phx_pools``).
"""

# Standard
from typing import Protocol, runtime_checkable

# First Party
from lmcache.integration.vllm.utils import get_size_bytes
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


@runtime_checkable
class DeviceMemoryPool(Protocol):
    """Backend-agnostic device memory pool for DMA-capable L1 tiers.

    A pool owns one device's worth of pre-allocated, DMA-registered device
    memory plus the backend-specific DMA handle. Backends implement this
    protocol; :class:`DeviceResidentL1MemoryManager` interacts only through this
    interface.

    A backend registers itself by adding an ``_init_<backend>_pools`` method
    to :class:`DeviceResidentL1MemoryManager` that populates ``_device_pools`` with
    concrete pool instances satisfying this protocol.
    """

    def allocate(
        self,
        *,
        shapes,
        dtypes,
        fmt,
    ) -> "MemoryObj | None":
        """Allocate one device MemoryObj.  None if pool exhausted."""
        ...

    def free(self, memory_obj: MemoryObj) -> None:
        """Free one device MemoryObj back to the pool."""
        ...

    def batched_free(self, memory_objs: list[MemoryObj]) -> None:
        """Free multiple device MemoryObjs."""
        ...

    def wait_for_available(self, required_bytes: int, timeout: float) -> bool:
        """Block until ≥ required_bytes free, or timeout.

        Returns True if space is available, False on timeout.
        """
        ...

    def get_free_bytes(self) -> int:
        """Return current free bytes in the pool."""
        ...


class DeviceResidentL1MemoryManager:
    """L1 memory manager for the device-resident tier.

    Holds device memory pools (one per configured device, behind the
    :class:`DeviceMemoryPool` protocol) and routes allocation/free by device.
    CPU objs continue through the normal CPU slab path.

    Backend-agnostic: specific pool implementations are selected by
    :attr:`DeviceResidentL1Config.backend` and instantiated in
    ``_init_<backend>_pools``. This class never imports backend types at
    module level — backends are registered as new ``_init_*_pools`` methods
    are added (a follow-up PR adds ``_init_phx_pools``).

    This manager is the sole ``_memory_manager`` instance of
    :class:`~lmcache.v1.distributed.l1_manager.L1Manager` when the
    device-resident tier is enabled. It serves **both** L1 tiers:

    - **CPU tier**: ``reserve_write`` (used by CPU-L1 adapters like
      redis/s3/disk, and by the store path via ``serde_wrapper``)
      allocates through the internal CPU :class:`L1MemoryManager`.
    - **Device tier**: ``device_reserve_write`` (used by device-L1
      adapters) allocates through :meth:`allocate_device`.

    :meth:`free` routes by ``obj.raw_tensor.device.type``: device objs go
    back to their parent :class:`DeviceMemoryPool`, CPU objs go back to
    the CPU slab.

    This manager satisfies :class:`L1ManagerProtocol` so that
    :class:`~lmcache.v1.distributed.l1_manager.L1Manager` can hold it behind
    one type.
    """

    def __init__(self, memory_config, device_resident_l1_config) -> None:
        """Create the manager.

        Args:
            memory_config: CPU memory config (for the internal CPU tier).
            device_resident_l1_config: Device-resident L1 configuration
                (backend, device_ids, buffer_size_mb, etc.).
        """
        # Lazy import to avoid circular dependency at module level
        # First Party
        from lmcache.v1.distributed.config import DeviceResidentL1Config
        from lmcache.v1.distributed.memory_manager.l1_memory_manager import (
            L1MemoryManager,
        )

        self._device_resident_l1_config: DeviceResidentL1Config = (
            device_resident_l1_config
        )
        # Single dict: dev_id → pool (pool internally holds DMA handle + base_ptr)
        self._device_pools: dict[int, DeviceMemoryPool] = {}
        # Validate backend + create device pools *before* allocating CPU
        # memory, so a misconfigured backend fails fast without wasting
        # resources.
        self._init_device_pools(device_resident_l1_config)
        # Hold a CPU L1 manager for the CPU tier: reserve_write (CPU-L1
        # adapters + store path) allocates through it, and free() routes
        # CPU objs back to it.
        self._cpu_manager = L1MemoryManager(memory_config)

    def _init_device_pools(self, config) -> None:
        """Dispatch to backend-specific pool initializer.

        This PR ships with no backends.
        New backends register here as ``elif`` branches.
        """
        backend = config.backend
        if backend == "phx":
            raise NotImplementedError(
                "PHX backend is implemented in a follow-up PR. "
                "This PR provides the framework only."
            )
        # elif backend == "gds":
        #     self._init_gds_pools(config)
        else:
            raise ValueError(f"Unsupported device-resident L1 backend: {backend!r}")

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate ``count`` CPU memory objects.

        This serves the **CPU tier**: ``L1Manager.reserve_write`` (used by
        CPU-L1 adapters and the store path) calls this method. Device
        allocation goes through :meth:`allocate_device` (which takes a
        ``kv_rank`` for device routing).

        Delegates to the internal CPU :class:`L1MemoryManager`.

        Args:
            layout_desc: Layout descriptor for the objects.
            count: Number of objects to allocate.

        Returns:
            ``(L1Error.SUCCESS, objects)`` on success, otherwise
            ``(L1Error.OUT_OF_MEMORY, [])``.
        """
        return self._cpu_manager.allocate(layout_desc, count)

    def allocate_device(
        self,
        layout_desc: MemoryLayoutDesc,
        count: int,
        kv_rank: int,
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate device-resident MemoryObjs for the given kv_rank.

        Routes ``kv_rank`` to a device id, then allocates from that device's
        pool. Implements backpressure: blocks until pool space is available
        or timeout.

        Args:
            layout_desc: Layout descriptor for the objects.
            count: Number of objects to allocate.
            kv_rank: Used to route to the correct device.

        Returns:
            ``(L1Error.SUCCESS, objects)`` on success.
            ``(L1Error.OUT_OF_MEMORY, [])`` on timeout/pool exhausted
            (all-or-nothing: partial allocation is freed before returning).
        """
        dev_id = self._kv_rank_to_device(kv_rank)
        pool = self._device_pools.get(dev_id)
        if pool is None:
            logger.warning(
                "DeviceResidentL1MemoryManager: no device pool for device %d "
                "(kv_rank=%d)",
                dev_id,
                kv_rank,
            )
            return L1Error.OUT_OF_MEMORY, []

        # Backpressure: wait for pool space before allocating
        size_per_obj = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        required = size_per_obj * count
        if not pool.wait_for_available(required, timeout=1.0):
            return L1Error.OUT_OF_MEMORY, []

        # Allocate one-by-one (pool may not have batched_allocate)
        objects: list[MemoryObj] = []
        for _ in range(count):
            obj = pool.allocate(
                shapes=layout_desc.shapes,
                dtypes=layout_desc.dtypes,
                fmt=None,
            )
            if obj is None:
                # Roll back partial allocation
                for o in objects:
                    pool.free(o)
                return L1Error.OUT_OF_MEMORY, []
            objects.append(obj)

        return L1Error.SUCCESS, objects

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Free a mix of CPU and device memory objects.

        This is the unified free path for both tiers: ``L1Manager`` calls
        this method for all frees (``finish_read``, ``delete``, ``clear``,
        ``close``, and OOM rollback). Routing is by
        ``obj.raw_tensor.device.type``:

        - Device objs (``!= "cpu"``): ``obj.parent().free(obj)`` — routes
          back to the :class:`DeviceMemoryPool` that allocated it. The
          pool's ``free`` notifies any ``wait_for_available`` waiter.
        - CPU objs: batched to ``self._cpu_manager.free()`` — returns to
          the CPU slab.

        Args:
            mem_objs: Objects to free (may be a mix of CPU and device).

        Returns:
            ``L1Error.SUCCESS``.
        """
        cpu_objs: list[MemoryObj] = []
        for o in mem_objs:
            if o is None:
                continue
            try:
                rt = o.raw_tensor
            except Exception:
                rt = None
            if rt is not None and rt.device.type != "cpu":
                parent = o.parent()
                if parent is not None:
                    parent.free(o)  # DeviceMemoryPool.free
                else:
                    logger.warning(
                        "DeviceResidentL1MemoryManager: device MemoryObj has no "
                        "parent allocator; cannot free, leaking device "
                        "memory"
                    )
            else:
                cpu_objs.append(o)
        if cpu_objs:
            self._cpu_manager.free(cpu_objs)
        return L1Error.SUCCESS

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Return the storage medium backing ``memory_obj``.

        Args:
            memory_obj: An object allocated by this manager.

        Returns:
            ``L1BackendType.DRAM`` for CPU objects,
            ``L1BackendType.DEVICE`` for device objects.
        """
        try:
            rt = memory_obj.raw_tensor
            if rt is not None and rt.device.type != "cpu":
                return L1BackendType.DEVICE
        except Exception:
            pass
        return L1BackendType.DRAM

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` for device pools.

        Only device pool usage is reported here. CPU tier usage is tracked
        separately by the internal ``_cpu_manager``.

        Note: this currently returns ``(0, sum_of_free_bytes)`` because the
        :class:`DeviceMemoryPool` protocol only exposes ``get_free_bytes``,
        not total capacity. Backend pools that track total capacity should
        override this method for accurate reporting.
        """
        total_used = 0
        total_capacity = 0
        for pool in self._device_pools.values():
            free = pool.get_free_bytes()
            # We don't track total per-pool directly; approximate via
            # used = total - free. Backend pools should override this if
            # they track total capacity.
            total_used += 0  # Updated by backend
            total_capacity += free
        return total_used, total_capacity

    def get_l1_memory_desc(self) -> "L1MemoryDesc | None":
        """Return ``None``: device pools are not a single registerable region.

        Device memory is owned by per-device pools behind the
        :class:`DeviceMemoryPool` protocol, not a single contiguous buffer
        that can be described by :class:`L1MemoryDesc`. P2P/NIXL transfer
        registration is therefore not supported for the device tier.
        """
        return None

    def close(self) -> None:
        """Release all device pool resources.

        Backend pools are responsible for their own cleanup (unregistering
        DMA mappings, closing device handles, etc.). The CPU manager is
        closed by :class:`L1Manager` separately.
        """
        for dev_id, pool in self._device_pools.items():
            if hasattr(pool, "close"):
                try:
                    pool.close()
                except Exception:
                    logger.warning(
                        "DeviceResidentL1MemoryManager: failed to close "
                        "pool for device %d",
                        dev_id,
                        exc_info=True,
                    )
        self._device_pools.clear()

    def memcheck(self) -> bool:
        """Check allocator consistency across all device pools.

        Logs the free bytes of each device pool. Returns ``True`` always
        (backends with leak detection should override this for real checks).
        """
        for dev_id, pool in self._device_pools.items():
            free = pool.get_free_bytes()
            logger.info(
                "DeviceResidentL1MemoryManager: device %d free bytes: %d",
                dev_id,
                free,
            )
        return True

    def _kv_rank_to_device(self, kv_rank: int) -> int:
        """Map a kv_rank to a device id.

        Default: ``kv_rank % num_devices``. Backends or configs may
        override this mapping. If no devices are configured, returns -1
        (which will cause ``allocate_device`` to return OOM).

        Args:
            kv_rank: The KV rank to map.

        Returns:
            The device id for this kv_rank.
        """
        device_ids = self._device_resident_l1_config.device_ids
        if not device_ids:
            return -1
        return device_ids[kv_rank % len(device_ids)]
