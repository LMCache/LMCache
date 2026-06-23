# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import abc

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.gating.write_veto import WriteVetoReason


class BaseStorageGate(abc.ABC):
    """
    Admission checks for storage operations (lookup / read / write / delete).

    The ``on_lookup``, ``on_read``, ``on_write``, and ``on_delete`` methods
    answer whether the corresponding operation may proceed. They MUST NOT mutate
    gate state (no counter updates and no other side effects). Backends update
    metrics via :meth:`record_lookup`, :meth:`record_read`, :meth:`record_write`,
    and :meth:`record_delete`.

    Extra eviction, pinning, and I/O belong in the backend, not in the gate.

    Note:
        Implementations may choose not to consult ``on_read`` on batched
        prefetch paths if rejecting reads would break list alignment with input
        keys; document behavior on the concrete backend.
    """

    @abc.abstractmethod
    def on_lookup(self, key: CacheEngineKey) -> bool:
        """
        Whether a successful metadata lookup (e.g. contains hit) is allowed.

        Must be side-effect free.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def on_read(self, key: CacheEngineKey) -> bool:
        """
        Whether a read/load of payload bytes for ``key`` may proceed.

        Must be side-effect free. Backends may omit this check on prefetch
        batches when results must match input key count.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def on_delete(self, key: CacheEngineKey) -> bool:
        """
        Whether removing ``key`` from storage may proceed.

        Must be side-effect free.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def explain_write_veto(
        self,
        key: CacheEngineKey,
        size_bytes: int,
    ) -> Optional[WriteVetoReason]:
        """
        If a write of ``size_bytes`` for ``key`` should be rejected, return why.

        Return a :class:`WriteVetoReason` for metrics; return ``None`` if allowed.
        Must be side-effect free and consistent with :meth:`on_write`.
        """
        raise NotImplementedError

    def on_write(self, key: CacheEngineKey, size_bytes: int) -> bool:
        """
        Whether persisting ``size_bytes`` for ``key`` may proceed.

        Default: ``explain_write_veto(...) is None``. Must be side-effect free.
        """
        return self.explain_write_veto(key, size_bytes) is None

    def record_lookup(self, key: CacheEngineKey) -> None:
        """Record a completed lookup hit (mutates internal counters if any)."""
        return None

    def record_read(self, key: CacheEngineKey) -> None:
        """Record a completed read/load (mutates internal counters if any)."""
        return None

    def record_write(self, key: CacheEngineKey, *, new_admission: bool = True) -> None:
        """
        Record a completed successful write (mutates internal counters if any).

        Args:
            key: Chunk key.
            new_admission: If True, count as a new on-disk admission; if False,
                only refresh counters used for read-based admission (e.g. reset
                read count after metadata refresh).
        """
        return None

    def record_delete(self, key: CacheEngineKey) -> None:
        """Record removal of ``key`` from storage (mutates internal counters if any)."""
        return None


class NullStorageGate(BaseStorageGate):
    """
    Gate that admits all operations and keeps no counters.
    """

    def on_lookup(self, key: CacheEngineKey) -> bool:  # noqa: ARG002
        return True

    def on_read(self, key: CacheEngineKey) -> bool:  # noqa: ARG002
        return True

    def on_delete(self, key: CacheEngineKey) -> bool:  # noqa: ARG002
        return True

    def explain_write_veto(
        self,
        key: CacheEngineKey,
        size_bytes: int,  # noqa: ARG002
    ) -> Optional[WriteVetoReason]:
        return None
