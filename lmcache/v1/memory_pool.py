# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, Optional, cast
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class PoolRequest:
    """
    Describes a single borrow request from the MemoryPool. Callers can either
    provide tensor shape/dtype/fmt or, for raw buffers, just the size.
    """

    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: MemoryFormat = MemoryFormat.KV_2LTD
    size: Optional[int] = None
    pinned: bool = False
    tag: str = "generic"
    eviction: bool = True
    busy_loop: bool = True

    def resolve_shape(self) -> torch.Size:
        if self.fmt == MemoryFormat.BINARY_BUFFER:
            if self.size is None and self.shape is None:
                raise ValueError("PoolRequest for BINARY_BUFFER expects size or shape")
            if self.shape is not None:
                return self.shape
            return torch.Size([self.size])  # type: ignore[arg-type]

        if self.shape is None or self.dtype is None:
            raise ValueError(
                "PoolRequest requires shape and dtype for tensor allocations"
            )
        return self.shape


class MemoryPool:
    """
    Thin facade that centralises engine-owned allocations and basic accounting.
    """

    _STAT_LIVE_LEASES = "pool_live_leases"
    _STAT_BORROWED_BYTES = "pool_borrowed_bytes"

    def __init__(
        self,
        allocator: MemoryAllocatorInterface,
        backend: Optional["AllocatorBackendInterface"] = None,
    ) -> None:
        self._allocator = allocator
        self._backend = backend
        self._live_leases = 0
        self._borrowed_bytes = 0
        self._lock = threading.Lock()

        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self._prometheus_logger = PrometheusLogger.GetInstanceOrNone()

    def borrow(self, req: PoolRequest) -> MemoryObj:
        """Borrow a MemoryObj from the underlying allocator."""
        shape = req.resolve_shape()
        memory_obj: Optional[MemoryObj]
        if self._backend is not None:
            allocate_from_pool = getattr(self._backend, "allocate_from_pool", None)
            if callable(allocate_from_pool):
                memory_obj = allocate_from_pool(req)
            else:
                backend_allocate = getattr(self._backend, "allocate", None)
                if callable(backend_allocate) and (
                    req.fmt != MemoryFormat.BINARY_BUFFER
                ):
                    if req.dtype is None:
                        raise ValueError(
                            "PoolRequest dtype must be set for backend allocations"
                        )
                    memory_obj = backend_allocate(
                        shape,
                        req.dtype,
                        req.fmt,
                        eviction=req.eviction,
                        busy_loop=req.busy_loop,
                    )
                else:
                    memory_obj = self._allocator.allocate(shape, req.dtype, req.fmt)
        else:
            memory_obj = self._allocator.allocate(shape, req.dtype, req.fmt)
        if memory_obj is None:
            raise RuntimeError(
                f"MemoryPool failed to allocate object (fmt={req.fmt}, tag={req.tag})"
            )

        memory_obj_any = cast(Any, memory_obj)
        original_parent = getattr(memory_obj_any, "parent_allocator", None)
        memory_obj_any._memory_pool_original_parent = original_parent
        memory_obj_any._memory_pool_released = False
        memory_obj_any.parent_allocator = self  # type: ignore[attr-defined]

        phy_size = memory_obj.meta.phy_size
        with self._lock:
            self._live_leases += 1
            self._borrowed_bytes += phy_size
            self._update_metrics()
        logger.debug(
            "MemoryPool borrowed %s bytes (tag=%s, fmt=%s, total_leases=%d)",
            phy_size,
            req.tag,
            req.fmt,
            self._live_leases,
        )
        return memory_obj

    def release(self, memory_obj: MemoryObj) -> None:
        memory_obj_any = cast(Any, memory_obj)
        phy_size = memory_obj.meta.phy_size
        already_released = getattr(memory_obj_any, "_memory_pool_released", False)
        if already_released:
            logger.warning(
                "MemoryPool.release called multiple times for %s", memory_obj
            )
            return
        memory_obj_any._memory_pool_released = True

        original_parent = getattr(
            memory_obj_any, "_memory_pool_original_parent", None
        )
        if original_parent is None or original_parent is self:
            original_parent = self._allocator

        # Restore the original parent allocator before delegating free.
        memory_obj_any.parent_allocator = original_parent  # type: ignore[attr-defined]
        try:
            original_parent.free(memory_obj)  # type: ignore[call-arg]
        finally:
            with self._lock:
                self._live_leases -= 1
                self._borrowed_bytes -= phy_size
                if self._live_leases < 0 or self._borrowed_bytes < 0:
                    logger.warning(
                        "MemoryPool counters negative after release "
                        "(leases=%d, bytes=%d)",
                        self._live_leases,
                        self._borrowed_bytes,
                    )
                    self._live_leases = max(self._live_leases, 0)
                    self._borrowed_bytes = max(self._borrowed_bytes, 0)
                self._update_metrics()
            if hasattr(memory_obj_any, "_memory_pool_original_parent"):
                delattr(memory_obj_any, "_memory_pool_original_parent")
        logger.debug(
            "MemoryPool released %s bytes (total_leases=%d)",
            phy_size,
            self._live_leases,
        )

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "live_leases": self._live_leases,
                "borrowed_bytes": self._borrowed_bytes,
            }

    def _update_metrics(self) -> None:
        update_custom_metric = getattr(
            self._stats_monitor, "update_custom_metric", None
        )
        if callable(update_custom_metric):
            update_custom_metric(self._STAT_LIVE_LEASES, self._live_leases)
            update_custom_metric(self._STAT_BORROWED_BYTES, self._borrowed_bytes)

        if self._prometheus_logger is not None:
            update_custom_gauge = getattr(
                self._prometheus_logger, "update_custom_gauge", None
            )
            if callable(update_custom_gauge):
                update_custom_gauge(self._STAT_LIVE_LEASES, self._live_leases)
                update_custom_gauge(self._STAT_BORROWED_BYTES, self._borrowed_bytes)

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """
        Allows MemoryObj instances to treat the pool as their parent allocator.
        """
        self.release(memory_obj)


@contextmanager
def lease(pool: MemoryPool, req: PoolRequest) -> Iterator[MemoryObj]:
    """
    Convenience helper that guarantees release even if the caller throws.
    """
    memory_obj = pool.borrow(req)
    try:
        yield memory_obj
    except Exception as exc:  # pragma: no cover - logged for observability
        logger.exception("MemoryPool lease raised exception: %s", exc)
        raise
    finally:
        pool.release(memory_obj)
