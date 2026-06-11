# SPDX-License-Identifier: Apache-2.0
"""Fault-injecting L2 adapter (test/diagnostic only).

A thin decorator that wraps a real inner L2 adapter (e.g. ``fs_native``) and
deterministically drops a subset of keys to simulate partial L2 retrieve
failures. Used to exercise the *segmented* code paths (gapped found-set ->
segmented prefetch -> segmented scatter / attention) that no normal workload
produces, since real caches return clean contiguous-prefix-then-tail results.

Two fault modes, mapping onto the adapter's two read primitives:

* ``miss``  -- clear dropped bits in ``query_lookup_and_lock_result``. The key
  is reported absent, so the matcher never selects it (a gap at lookup). The
  inner adapter already locked the key during lookup, so the hidden keys are
  unlocked here to avoid leaking a read lock.
* ``error`` -- clear dropped bits in ``query_load_result``. Lookup reported the
  key present but the load fails (the faithful "L2 retrieve error"); the
  prefetch controller releases the load-failed read locks itself via the trim
  mask, so no manual unlock is needed.
* ``both``  -- apply to both (a key dropped at lookup is also dropped at load).

The drop-set is deterministic: a stable hash of the key bucketed by ``rate``
(so lookup and load agree within a request and runs reproduce), plus an
optional ``gap_indices`` set of exact task-positions to drop for precise
single-gap repros.

Limitations (test scope): serde / eviction configured on this adapter's JSON
spec are not propagated to the inner adapter (the inner is built directly by
this adapter's factory, bypassing the StorageManager's per-adapter wrapping).
Configure the inner adapter for a plain store and keep faults at this layer.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Optional
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

_VALID_MODES = ("miss", "error", "both")
_HASH_DENOM = 1_000_000


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


class FaultInjectL2AdapterConfig(L2AdapterConfigBase):
    """Config for the fault-injecting L2 adapter.

    JSON fields:
    - inner (dict, required): full adapter spec for the wrapped adapter, with
      its own "type" (e.g. {"type": "fs_native", "base_path": "/dev/shm/x"}).
    - mode (str): "miss" | "error" | "both". Default "error".
    - rate (float): per-key drop probability in [0, 1]. Default 0.0 (pass-through).
    - seed (int): seed for the deterministic per-key hash. Default 0.
    - gap_indices (list[int]): exact task-positions to always drop (in addition
      to rate-based drops). Default [] -- mainly for precise unit tests.
    """

    def __init__(
        self,
        inner_config: L2AdapterConfigBase,
        mode: str,
        rate: float,
        seed: int,
        gap_indices: tuple[int, ...],
    ) -> None:
        self.inner_config = inner_config
        self.mode = mode
        self.rate = rate
        self.seed = seed
        self.gap_indices = gap_indices

    @classmethod
    def from_dict(cls, d: dict) -> "FaultInjectL2AdapterConfig":
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

        mode = d.get("mode", "error")
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")

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

        return cls(
            inner_config=inner_config,
            mode=mode,
            rate=float(rate),
            seed=seed,
            gap_indices=tuple(raw_gap),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Fault-injecting L2 adapter (test only). Fields:\n"
            "- inner (dict, required): wrapped adapter spec, e.g. "
            '{"type":"fs_native","base_path":"/dev/shm/x"}\n'
            "- mode (str): 'miss' | 'error' | 'both' (default 'error')\n"
            "- rate (float): per-key drop probability in [0,1] (default 0.0)\n"
            "- seed (int): deterministic hash seed (default 0)\n"
            "- gap_indices (list[int]): exact task-positions to always drop"
        )


# -----------------------------------------------------------------------------
# Adapter (decorator)
# -----------------------------------------------------------------------------


class FaultInjectL2Adapter(L2AdapterInterface):
    """Decorator over an inner L2 adapter that drops a deterministic key subset.

    All operations delegate to ``inner``; only the two read-result queries are
    post-processed (bits cleared) according to ``mode``.
    """

    def __init__(
        self,
        inner: L2AdapterInterface,
        mode: str,
        rate: float,
        seed: int,
        gap_indices: tuple[int, ...],
    ) -> None:
        super().__init__(max_capacity_bytes=0)
        self._inner = inner
        self._mode = mode
        self._rate = rate
        self._seed = seed
        self._gap_indices = frozenset(gap_indices)
        # task_id -> keys, so query_*_result can map positions back to keys.
        self._lookup_keys: dict[L2TaskId, list[ObjectKey]] = {}
        self._load_keys: dict[L2TaskId, list[ObjectKey]] = {}
        self._keys_lock = threading.Lock()
        logger.warning(
            "FaultInjectL2Adapter ACTIVE (mode=%s rate=%.3f seed=%d gap_indices=%s) "
            "wrapping %s -- test/diagnostic use only.",
            mode,
            rate,
            seed,
            sorted(self._gap_indices),
            type(inner).__name__,
        )

    # -- drop decision --------------------------------------------------------

    def _should_drop_key(self, key: ObjectKey) -> bool:
        if self._rate <= 0.0:
            return False
        h = hashlib.blake2b(f"{self._seed}:{key!r}".encode(), digest_size=8).digest()
        bucket = int.from_bytes(h, "big") % _HASH_DENOM
        return bucket < int(self._rate * _HASH_DENOM)

    def _drop_positions(self, keys: list[ObjectKey]) -> list[int]:
        dropped = []
        for i, key in enumerate(keys):
            if i in self._gap_indices or self._should_drop_key(key):
                dropped.append(i)
        return dropped

    # -- event fds (delegate) -------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._inner.get_store_event_fd()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._inner.get_lookup_and_lock_event_fd()

    def get_load_event_fd(self) -> int:
        return self._inner.get_load_event_fd()

    # -- store (delegate) -----------------------------------------------------

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        return self._inner.submit_store_task(keys, objects)

    def pop_completed_store_tasks(self) -> dict[L2TaskId, L2StoreResult]:
        return self._inner.pop_completed_store_tasks()

    # -- lookup and lock (intercept for 'miss') -------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        task_id = self._inner.submit_lookup_and_lock_task(keys)
        with self._keys_lock:
            self._lookup_keys[task_id] = keys
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        bitmap = self._inner.query_lookup_and_lock_result(task_id)
        if bitmap is None:
            return None
        with self._keys_lock:
            keys = self._lookup_keys.pop(task_id, None)
        if self._mode in ("miss", "both") and keys is not None:
            dropped = self._drop_positions(keys)
            if dropped:
                # Clear hidden bits and release the locks the inner adapter took
                # for those keys during lookup-and-lock (else they leak).
                hidden_keys = [keys[i] for i in dropped if bitmap.test(i)]
                for i in dropped:
                    bitmap.clear(i)
                if hidden_keys:
                    self._inner.submit_unlock(hidden_keys)
                logger.debug(
                    "FaultInject miss: task %s dropped %d/%d lookup keys",
                    task_id,
                    len(dropped),
                    len(keys),
                )
        return bitmap

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        self._inner.submit_unlock(keys)

    # -- load (intercept for 'error') -----------------------------------------

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        task_id = self._inner.submit_load_task(keys, objects)
        with self._keys_lock:
            self._load_keys[task_id] = keys
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        bitmap = self._inner.query_load_result(task_id)
        if bitmap is None:
            return None
        with self._keys_lock:
            keys = self._load_keys.pop(task_id, None)
        if self._mode in ("error", "both") and keys is not None:
            dropped = self._drop_positions(keys)
            for i in dropped:
                bitmap.clear(i)
            if dropped:
                logger.debug(
                    "FaultInject error: task %s dropped %d/%d load keys",
                    task_id,
                    len(dropped),
                    len(keys),
                )
        return bitmap

    # -- listener / eviction / usage (delegate) -------------------------------

    def register_listener(self, listener: L2AdapterListener) -> None:
        self._inner.register_listener(listener)

    def delete(self, keys: list[ObjectKey]) -> None:
        self._inner.delete(keys)

    def get_usage(self) -> AdapterUsage:
        return self._inner.get_usage()

    @property
    def supports_global_eviction(self) -> bool:
        return self._inner.supports_global_eviction

    def report_status(self) -> dict:
        status = self._inner.report_status()
        status["fault_inject"] = {
            "mode": self._mode,
            "rate": self._rate,
            "seed": self._seed,
            "gap_indices": sorted(self._gap_indices),
        }
        return status

    def close(self) -> None:
        with self._keys_lock:
            self._lookup_keys.clear()
            self._load_keys.clear()
        self._inner.close()


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

register_l2_adapter_type("fault_inject", FaultInjectL2AdapterConfig)


def _create_fault_inject_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Build the inner adapter from the registry, then wrap it."""
    # First Party
    from lmcache.v1.distributed.l2_adapters.factory import (  # noqa: PLC0415
        create_l2_adapter_from_registry,
    )

    assert isinstance(config, FaultInjectL2AdapterConfig)
    inner = create_l2_adapter_from_registry(config.inner_config, l1_memory_desc)
    return FaultInjectL2Adapter(
        inner,
        mode=config.mode,
        rate=config.rate,
        seed=config.seed,
        gap_indices=config.gap_indices,
    )


register_l2_adapter_factory("fault_inject", _create_fault_inject_adapter)
