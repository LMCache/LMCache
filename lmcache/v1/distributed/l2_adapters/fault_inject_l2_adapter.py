# SPDX-License-Identifier: Apache-2.0
"""
Fault-injecting L2 adapter (test/diagnostic only).

Wraps a real inner L2 adapter (e.g. ``fs_native``) and deterministically
drops keys to simulate partial L2 retrieve failures, exercising the
segmented code paths (gapped found-set -> segmented prefetch ->
segmented scatter/attention) that real caches never produce.

Faults the load read primitive only: a dropped key is reported present
at lookup but its load fails (the faithful "L2 retrieve error"; the
prefetch controller releases the load-failed locks via the trim mask).
The lookup-miss case is intentionally not modeled -- a key absent at
lookup merely shortens the found-set, which the PREFIX trim policy
(count_leading_ones) already covers. The drop-set is a stable hash of
the key bucketed by ``rate``, plus optional ``gap_indices`` for precise
single-gap repros.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import hashlib
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import L1MemoryDesc

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.internal_api import L2AdapterListener, L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    AdapterUsage,
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

_HASH_DENOM = 1_000_000


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


class FaultInjectL2AdapterConfig(L2AdapterConfigBase):
    """Config for the fault-injecting L2 adapter.

    JSON fields:
    - inner (dict, required): full adapter spec for the wrapped adapter, with
      its own "type" (e.g. {"type": "fs_native", "base_path": "/dev/shm/x"}).
    - rate (float): per-key drop probability in [0, 1]. Default 0.0 (pass-through).
    - seed (int): seed for the deterministic per-key hash. Default 0.
    - gap_indices (list[int]): exact task-positions to always drop (in addition
      to rate-based drops). Default [] -- mainly for precise unit tests.
    - drop_chunk_hashes (list[str]): hex-encoded chunk hashes to always drop,
      matched on ObjectKey.chunk_hash. Content-keyed, so the SAME chunk fails in
      every load task (prefix + sparse legs) -- a persistent single-gap repro.
      Default [].
    """

    def __init__(
        self,
        inner_config: L2AdapterConfigBase,
        rate: float,
        seed: int,
        gap_indices: tuple[int, ...],
        drop_chunk_hashes: tuple[str, ...] = (),
    ) -> None:
        self.inner_config = inner_config
        self.rate = rate
        self.seed = seed
        self.gap_indices = gap_indices
        self.drop_chunk_hashes = drop_chunk_hashes

    @classmethod
    def from_dict(cls, d: dict) -> "FaultInjectL2AdapterConfig":
        """Parse a fault-inject adapter spec, building the inner config too.

        The ``"inner"`` sub-dict is dispatched through the L2 adapter config
        registry (lazy-importing its module if needed), mirroring
        ``parse_args_to_l2_adapters_config`` so the inner adapter's own
        eviction / persist / serde keys are honored.

        Args:
            d: The adapter spec dict (see the class docstring for fields).

        Returns:
            A populated ``FaultInjectL2AdapterConfig``.

        Raises:
            ValueError: If ``inner`` is missing or malformed, the inner type
                is unknown, or ``rate`` / ``seed`` / ``gap_indices`` fail
                validation.
        """
        inner = d.get("inner")
        if not isinstance(inner, dict):
            raise ValueError("'inner' must be an adapter spec dict with a 'type' field")
        inner_type = inner.get("type")
        if not isinstance(inner_type, str):
            raise ValueError("'inner' adapter spec must include a string 'type'")

        # Build the inner adapter config via the registry (lazy-importing its
        # module if needed), mirroring parse_args_to_l2_adapters_config.
        # First Party
        from lmcache.v1.distributed.l2_adapters.config import (  # noqa: PLC0415
            _L2_ADAPTER_CONFIG_REGISTRY,
            _ensure_config_loaded,
        )

        _ensure_config_loaded(inner_type)
        if inner_type not in _L2_ADAPTER_CONFIG_REGISTRY:
            raise ValueError(f"unknown inner adapter type {inner_type!r}")
        inner_cls = _L2_ADAPTER_CONFIG_REGISTRY[inner_type]
        inner_config = inner_cls.from_dict(inner)
        inner_config.eviction_config = cls._parse_eviction_config(inner)
        inner_config.persist_config = cls._parse_persist_config(inner)
        inner_config.serde_config = cls._parse_serde_config(inner)

        rate = d.get("rate", 0.0)
        if not isinstance(rate, (int, float)) or not (0.0 <= rate <= 1.0):
            raise ValueError("rate must be a number in [0, 1]")

        seed = d.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")

        raw_gap = d.get("gap_indices", [])
        if not isinstance(raw_gap, list) or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in raw_gap
        ):
            raise ValueError("gap_indices must be a list of non-negative integers")

        raw_hashes = d.get("drop_chunk_hashes", [])
        if not isinstance(raw_hashes, list) or any(
            not isinstance(h, str) for h in raw_hashes
        ):
            raise ValueError("drop_chunk_hashes must be a list of hex strings")
        for h in raw_hashes:
            try:
                bytes.fromhex(h)
            except ValueError as e:
                raise ValueError(f"drop_chunk_hashes entry {h!r} is not hex") from e

        return cls(
            inner_config=inner_config,
            rate=float(rate),
            seed=seed,
            gap_indices=tuple(raw_gap),
            drop_chunk_hashes=tuple(raw_hashes),
        )

    @classmethod
    def help(cls) -> str:
        """Return a help string describing this adapter's JSON config fields."""
        return (
            "Fault-injecting L2 adapter (test only; drops keys at load to "
            "simulate L2 retrieve errors). Fields:\n"
            "- inner (dict, required): wrapped adapter spec, e.g. "
            '{"type":"fs_native","base_path":"/dev/shm/x"}\n'
            "- rate (float): per-key drop probability in [0,1] (default 0.0)\n"
            "- seed (int): deterministic hash seed (default 0)\n"
            "- gap_indices (list[int]): exact task-positions to always drop\n"
            "- drop_chunk_hashes (list[str]): hex chunk hashes to always drop "
            "(content-keyed; persistent across load legs)"
        )


# -----------------------------------------------------------------------------
# Adapter (decorator)
# -----------------------------------------------------------------------------


class FaultInjectL2Adapter(L2AdapterInterface):
    """Decorator over an inner L2 adapter that drops a deterministic key subset.

    All operations delegate to ``inner``; only the load-result query is
    post-processed (dropped bits cleared) to simulate L2 retrieve errors.
    """

    def __init__(
        self,
        inner: L2AdapterInterface,
        rate: float,
        seed: int,
        gap_indices: tuple[int, ...],
        drop_chunk_hashes: tuple[str, ...] = (),
    ) -> None:
        """Wrap ``inner`` with deterministic load-failure injection.

        Args:
            inner: The real L2 adapter that every operation delegates to.
            rate: Per-key drop probability in [0, 1]. ``0`` disables
                rate-based drops (``gap_indices`` still applies).
            seed: Seed for the deterministic per-key hash, so runs reproduce.
            gap_indices: Exact task-positions to always drop, in addition to
                the rate-based drops.
            drop_chunk_hashes: Hex-encoded chunk hashes to always drop, matched
                on ``ObjectKey.chunk_hash``. Content-keyed, so the same chunk
                fails in every load task (a persistent single-gap repro).
        """
        # Usage and eviction are delegated to the inner adapter, so the base
        # class's byte accounting (capacity 0) is intentionally unused here.
        super().__init__(max_capacity_bytes=0)
        self._inner = inner
        self._rate = rate
        self._seed = seed
        self._gap_indices = frozenset(gap_indices)
        self._drop_hash_bytes = frozenset(bytes.fromhex(h) for h in drop_chunk_hashes)
        # task_id -> load keys, so query_load_result can map a dropped bit
        # position back to its key.
        self._load_keys: dict[L2TaskId, list[ObjectKey]] = {}
        self._keys_lock = threading.Lock()
        logger.warning(
            "FaultInjectL2Adapter ACTIVE (rate=%.3f seed=%d gap_indices=%s "
            "drop_chunk_hashes=%d) wrapping %s -- test/diagnostic use only.",
            rate,
            seed,
            sorted(self._gap_indices),
            len(self._drop_hash_bytes),
            type(inner).__name__,
        )

    # -- drop decision --------------------------------------------------------

    def _should_drop_key(self, key: ObjectKey) -> bool:
        """Return whether ``key`` falls in the rate-based drop bucket.

        Deterministic in ``(seed, key)``: a stable blake2b hash bucketed by
        ``rate``, so a key drops (or not) identically at lookup and at load
        within a request, and reproducibly across runs with the same seed.
        Returns ``False`` immediately when ``rate <= 0`` (pass-through).
        """
        if self._rate <= 0.0:
            return False
        h = hashlib.blake2b(f"{self._seed}:{key!r}".encode(), digest_size=8).digest()
        bucket = int.from_bytes(h, "big") % _HASH_DENOM
        return bucket < int(self._rate * _HASH_DENOM)

    def _drop_positions(self, keys: list[ObjectKey]) -> list[int]:
        """Return the positions in ``keys`` to drop, in ascending order.

        A position is dropped if its key's ``chunk_hash`` is in
        ``drop_chunk_hashes`` (content-keyed, persistent across legs), or it is
        listed in ``gap_indices`` (an exact, always-drop position), or its key
        falls in the rate-based bucket per ``_should_drop_key``.
        """
        dropped = []
        for i, key in enumerate(keys):
            if (
                key.chunk_hash in self._drop_hash_bytes
                or i in self._gap_indices
                or self._should_drop_key(key)
            ):
                dropped.append(i)
        return dropped

    # -- event fds (delegate) -------------------------------------------------

    def get_store_event_fd(self) -> int:
        """Return the inner adapter's store event fd."""
        return self._inner.get_store_event_fd()

    def get_lookup_and_lock_event_fd(self) -> int:
        """Return the inner adapter's lookup-and-lock event fd."""
        return self._inner.get_lookup_and_lock_event_fd()

    def get_load_event_fd(self) -> int:
        """Return the inner adapter's load event fd."""
        return self._inner.get_load_event_fd()

    # -- store (delegate) -----------------------------------------------------

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Delegate the store task to the inner adapter (store is never faulted)."""
        return self._inner.submit_store_task(keys, objects)

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        """Delegate to the inner adapter; store results are passed through."""
        return self._inner.pop_completed_store_tasks()

    # -- lookup and lock (pure delegation) ------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        """Delegate the lookup-and-lock task to the inner adapter (not faulted)."""
        return self._inner.submit_lookup_and_lock_task(keys)

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Delegate to the inner adapter; lookup results are passed through.

        Only the load primitive is faulted, so lookup reports keys present as
        usual; the gap appears later when their load fails.
        """
        return self._inner.query_lookup_and_lock_result(task_id)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        """Delegate the unlock to the inner adapter."""
        self._inner.submit_unlock(keys)

    # -- load (intercept for 'error') -----------------------------------------

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Submit a load task, recording ``keys`` for result mapping.

        Thread-safe. The key list is stored under the returned task id so
        ``query_load_result`` can map dropped bit positions back to keys.
        """
        task_id = self._inner.submit_load_task(keys, objects)
        with self._keys_lock:
            self._load_keys[task_id] = keys
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        """Query the load result, clearing the dropped keys' bits.

        Delegates to the inner adapter, then clears the bits at the dropped
        positions. This is the faithful "L2 retrieve error": lookup reported
        the key present but the load fails. No unlock is needed here; the
        prefetch controller releases the load-failed read locks itself via the
        trim mask.

        Thread-safe. Non-idempotent: returns a non-None bitmap only once per
        ``task_id``.

        Args:
            task_id: The task id returned by ``submit_load_task``.

        Returns:
            The post-processed bitmap, or ``None`` if the inner task has not
            completed yet.
        """
        bitmap = self._inner.query_load_result(task_id)
        if bitmap is None:
            return None
        with self._keys_lock:
            keys = self._load_keys.pop(task_id, None)
        if keys is not None:
            dropped = self._drop_positions(keys)
            for i in dropped:
                bitmap.clear(i)
            if dropped:
                logger.debug(
                    "FaultInject: task %s dropped %d/%d load keys",
                    task_id,
                    len(dropped),
                    len(keys),
                )
        return bitmap

    # -- listener / eviction / usage (delegate) -------------------------------

    def register_listener(self, listener: L2AdapterListener) -> None:
        """Forward the listener registration to the inner adapter."""
        self._inner.register_listener(listener)

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delegate the delete to the inner adapter."""
        self._inner.delete(keys)

    def get_usage(self) -> AdapterUsage:
        """Return the inner adapter's usage; this layer holds no data of its own."""
        return self._inner.get_usage()

    @property
    def supports_global_eviction(self) -> bool:
        """Whether the inner adapter supports aggregate usage-based eviction."""
        return self._inner.supports_global_eviction

    def report_status(self) -> dict:
        """Return the inner adapter's status, annotated with the fault config.

        Adds a ``"fault_inject"`` sub-dict (rate, seed, gap_indices) so the
        active fault configuration is visible in diagnostics.
        """
        status = self._inner.report_status()
        status["fault_inject"] = {
            "rate": self._rate,
            "seed": self._seed,
            "gap_indices": sorted(self._gap_indices),
            "drop_chunk_hashes": sorted(h.hex() for h in self._drop_hash_bytes),
        }
        return status

    def close(self) -> None:
        """Clear the per-task key map, then close the inner adapter."""
        with self._keys_lock:
            self._load_keys.clear()
        self._inner.close()


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

register_l2_adapter_type("fault_inject", FaultInjectL2AdapterConfig)


def _create_fault_inject_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "L1MemoryDesc | None" = None,
) -> L2AdapterInterface:
    """Build the inner adapter from the registry, then wrap it for faults.

    Matches the L2 adapter factory signature so it can be registered under
    the ``"fault_inject"`` type name.

    Args:
        config: A ``FaultInjectL2AdapterConfig`` whose ``inner_config`` names
            the wrapped adapter.
        l1_memory_desc: L1 memory descriptor forwarded to the inner adapter's
            factory (some adapters need it for aligned buffers); ``None`` when
            not applicable.

    Returns:
        A ``FaultInjectL2Adapter`` wrapping the freshly built inner adapter.
    """
    # First Party
    from lmcache.v1.distributed.l2_adapters.factory import (  # noqa: PLC0415
        create_l2_adapter_from_registry,
    )

    assert isinstance(config, FaultInjectL2AdapterConfig)
    inner = create_l2_adapter_from_registry(config.inner_config, l1_memory_desc)
    return FaultInjectL2Adapter(
        inner,
        rate=config.rate,
        seed=config.seed,
        gap_indices=config.gap_indices,
        drop_chunk_hashes=config.drop_chunk_hashes,
    )


register_l2_adapter_factory("fault_inject", _create_fault_inject_adapter)
