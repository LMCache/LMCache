# SPDX-License-Identifier: Apache-2.0
# Future
from __future__ import annotations

# Standard
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, Iterator, Optional, cast
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
)

logger = init_logger(__name__)


@dataclass(frozen=True, slots=True)
class PoolRequest:
    """
    Describes a single borrow request from the MemoryPool. Callers can either
    provide tensor shape/dtype/fmt or, for raw buffers, just the size.
    """

    shape: Optional[torch.Size] = None
    dtype: Optional[torch.dtype] = None
    fmt: MemoryFormat = MemoryFormat.BINARY_BUFFER
    size: Optional[int] = None
    tag: str = "generic"
    eviction: bool = True
    busy_loop: bool = False

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
    Thin facade that centralises engine-owned allocations and ref-count hygiene.

    Responsibilities:
      * Delegate allocations to either the backend hook or the raw allocator.
      * Swap in the pool as the parent allocator so ref_count_down() leads back here.
      * Keep lightweight counters for observability/debug logging.
    """

    def __init__(
        self,
        allocator: MemoryAllocatorInterface,
        borrow_hook: Optional[Callable[[PoolRequest], Optional[MemoryObj]]] = None,
    ) -> None:
        self._allocator = allocator
        self._borrow_hook = borrow_hook
        self._live_leases = 0
        self._borrowed_bytes = 0
        self._lock = threading.Lock()

    def borrow(self, req: PoolRequest) -> MemoryObj:
        """Borrow a MemoryObj from the underlying allocator."""
        shape = req.resolve_shape()
        memory_obj: Optional[MemoryObj] = None
        if self._borrow_hook is not None:
            memory_obj = self._borrow_hook(req)
        if memory_obj is None:
            if req.dtype is None and req.fmt != MemoryFormat.BINARY_BUFFER:
                raise ValueError("dtype must be specified for tensor allocations")
            memory_obj = self._allocator.allocate(shape, req.dtype, req.fmt)
        if memory_obj is None:
            raise RuntimeError(
                f"MemoryPool failed to allocate object (fmt={req.fmt}, tag={req.tag})"
            )

        self._attach_parent_allocator(memory_obj)
        phy_size = memory_obj.meta.phy_size
        self._record_borrow(phy_size)
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
        original_parent = getattr(memory_obj_any, "_memory_pool_original_parent", None)
        if original_parent is None or original_parent is self:
            original_parent = self._allocator

        # Restore the original parent allocator before delegating free.
        memory_obj_any.parent_allocator = original_parent  # type: ignore[attr-defined]
        try:
            original_parent.free(memory_obj)  # type: ignore[call-arg]
        finally:
            self._record_release(phy_size)
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

    def lease(self, req: PoolRequest) -> ContextManager[MemoryObj]:
        """
        Context manager that borrows and automatically releases the request.
        """

        @contextmanager
        def _ctx() -> Iterator[MemoryObj]:
            memory_obj = self.borrow(req)
            try:
                yield memory_obj
            except Exception as exc:
                logger.exception("MemoryPool lease raised exception: %s", exc)
                raise
            finally:
                self.release(memory_obj)

        return _ctx()

    def free(self, memory_obj: MemoryObj, allocator_type: Optional[str] = None) -> None:
        """
        Allows MemoryObj instances to treat the pool as their parent allocator.
        """
        self.release(memory_obj)

    def _attach_parent_allocator(self, memory_obj: MemoryObj) -> None:
        """
        Remember the original parent allocator and redirect ref_count_down to the pool.
        """
        memory_obj_any = cast(Any, memory_obj)
        if getattr(memory_obj_any, "_memory_pool_released", False):
            # If borrowed after a previous release, clear book-keeping first.
            delattr(memory_obj_any, "_memory_pool_released")
        original_parent = getattr(memory_obj_any, "parent_allocator", None)
        memory_obj_any._memory_pool_original_parent = original_parent
        memory_obj_any._memory_pool_released = False
        memory_obj_any.parent_allocator = self  # type: ignore[attr-defined]

    def _record_borrow(self, bytes_borrowed: int) -> None:
        """Track a successful borrow in a threadsafe way."""
        with self._lock:
            self._live_leases += 1
            self._borrowed_bytes += bytes_borrowed

    def _record_release(self, bytes_released: int) -> None:
        """Track a release while clamping counters at zero."""
        with self._lock:
            self._live_leases = max(0, self._live_leases - 1)
            self._borrowed_bytes = max(0, self._borrowed_bytes - bytes_released)


@contextmanager
def lease(pool: MemoryPool, req: PoolRequest) -> Iterator[MemoryObj]:
    """
    Convenience helper that guarantees release even if the caller throws.
    """
    with pool.lease(req) as memory_obj:
        yield memory_obj
