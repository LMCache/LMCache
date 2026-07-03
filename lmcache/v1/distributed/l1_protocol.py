# SPDX-License-Identifier: Apache-2.0

"""Structural ``Protocol`` for the L1 control-state manager.

The seam that lets the stock controllers drive either ``L1Manager`` or the
CXL-backed ``MaruL1Manager``; structural (no inheritance), so ``l1_manager.py``
stays untouched. The ``Fires ...`` line in each docstring is the listener-event
contract both backends must honor; see ``L1Manager`` for full error semantics.
"""

# Standard
from typing import Any, Literal, Protocol, runtime_checkable

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerListener, L1MemoryDesc
from lmcache.v1.distributed.l1_manager import L1ObjectState, L1OperationResult


@runtime_checkable
class L1ManagerInterface(Protocol):
    """L1 control surface shared by ``L1Manager`` and ``MaruL1Manager``."""

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for the ``on_l1_keys_*`` events."""
        ...

    def reserve_read(
        self, keys: list[ObjectKey], extra_count: int = 0
    ) -> dict[ObjectKey, L1OperationResult]:
        """Reserve read; ``1+extra_count`` holds/key.
        Fires ``on_l1_keys_reserved_read``."""
        ...

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Return read-locked objects without new locks
        (between reserve_read and finish_read)."""
        ...

    def finish_read(
        self, keys: list[ObjectKey], extra_count: int = 0
    ) -> dict[ObjectKey, L1Error]:
        """Release ``1+extra_count`` holds/key.
        Fires ``on_l1_keys_read_finished`` (+ ``on_l1_keys_deleted_by_manager`` for
        temporaries dropped at count 0)."""
        ...

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        mode: Literal["new", "update", "all"] = "all",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Allocate + write-lock buffers; ``is_temporary[i]`` drops key i after read.
        Fires ``on_l1_keys_reserved_write``."""
        ...

    def finish_write(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Release write locks.
        Fires ``on_l1_keys_write_finished`` (write-through trigger)."""
        ...

    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey], extra_count: int = 0
    ) -> dict[ObjectKey, L1OperationResult]:
        """Finish write + take ``1+extra_count`` read holds/key (L2->L1 promote).
        Fires ``on_l1_keys_finish_write_and_reserve_read`` (NOT write_finished)."""
        ...

    def delete(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Delete unlocked keys (locked keys refused).
        Fires ``on_l1_keys_deleted_by_manager`` for keys actually removed."""
        ...

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Mark keys accessed.
        Fires ``on_l1_keys_accessed``."""
        ...

    def clear(self, force: bool = False) -> None:
        """Free objects (``force`` frees locked too).
        Fires ``on_l1_keys_deleted_by_manager`` for freed keys."""
        ...

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Whether ``key`` exists and is unlocked (lock-free)."""
        ...

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)`` of the L1 medium."""
        ...

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return the L1 buffer descriptor, or ``None`` if not exposed."""
        ...

    def close(self) -> None:
        """Free all objects and release resources."""
        ...

    def report_status(self) -> dict[str, Any]:
        """Return a status dict of L1 cache state."""
        ...

    def get_object_state(self, key: ObjectKey) -> L1ObjectState | None:
        """Return the internal state of ``key``, or ``None`` if absent."""
        ...

    def memcheck(self) -> bool:
        """Run the medium's memory consistency check."""
        ...
