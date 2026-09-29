# SPDX-License-Identifier: Apache-2.0
"""
Distributed multi-tier storage manager for MP mode
"""

# Standard
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator, Optional
import threading
import time
import weakref

# First Party
from lmcache.lmcache_native import PeriodicEventNotifier
from lmcache.logging import init_logger
from lmcache.utils import lmcache_deprecate
from lmcache.v1.distributed.api import (
    CapacitySnapshot,
    L1BackendType,
    MemoryLayoutDesc,
    ModuleMemoryCapacity,
    ObjectKey,
    PrefetchHandle,
    PrefetchResult,
    PrefetchTaskSpec,
    Tier,
)
from lmcache.v1.distributed.bitmap_ops import fold_unfold_grouped
from lmcache.v1.distributed.config import (
    EvictionConfig,
    StorageManagerConfig,
    get_configured_capacity_bytes,
)
from lmcache.v1.distributed.error import L1Error, strerror
from lmcache.v1.distributed.internal_api import L1MemoryDesc, L2AdapterListener
from lmcache.v1.distributed.l1_manager import L1Manager, L1OperationResult
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage, L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
from lmcache.v1.distributed.l2_adapters.reconfiguration import (
    L2ReconfigurableAdapter,
    L2ReconfigureError,
)
from lmcache.v1.distributed.l2_adapters.serde_wrapper import SerdeL2AdapterWrapper
from lmcache.v1.distributed.quota_manager import QuotaManager
from lmcache.v1.distributed.serde import create_serde_processor
from lmcache.v1.distributed.storage_controllers import (
    L1EvictionController,
    L2AdapterEvictionState,
    L2EvictionController,
    PrefetchController,
    StoreController,
)
from lmcache.v1.distributed.storage_controllers.prefetch_policy import (
    create_prefetch_policy,
)
from lmcache.v1.distributed.storage_controllers.store_policy import (
    create_store_policy,
)
from lmcache.v1.distributed.storage_controllers.utils import (
    L1ManagerDescriptor,
    L2AdapterDescriptor,
)
from lmcache.v1.distributed.storage_controllers.write_policy import (
    DefaultWritePolicy,
    WritePolicy,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.mp_observability.trace.decorator import (
    enable_tracing,
    is_tracing_enabled,
    publish_call_event,
)
from lmcache.v1.platform import HAS_EVENTFD

logger = init_logger(__name__)

# L1 write tag for every object reserved through this manager. Sharing one
# tag makes concurrent stores of the same key exclude each other.
_L1_WRITE_TAG = "storage_manager"


class StorageManager:
    def __init__(
        self, config: StorageManagerConfig, write_policy: WritePolicy | None = None
    ) -> None:
        """Create peer L1 managers; write_policy selects serving-engine placement."""
        self._l1_configs = config.l1_manager_configs
        self._l1_managers = [L1Manager(c) for c in self._l1_configs]
        self._l1_descriptors = [
            L1ManagerDescriptor(index=i, config=c)
            for i, c in enumerate(self._l1_configs)
        ]
        self._l1_by_tag = {c.tag: i for i, c in enumerate(self._l1_configs)}
        self._write_policy = write_policy or DefaultWritePolicy()
        self._write_lock = threading.Lock()
        self._write_locations: weakref.WeakValueDictionary[ObjectKey, MemoryObj] = (
            weakref.WeakValueDictionary()
        )
        self._object_owners: weakref.WeakKeyDictionary[MemoryObj, int] = (
            weakref.WeakKeyDictionary()
        )
        # The public transfer API identifies reads by key. Keep concurrent
        # prefetches on the same readable copy even when L1s contain duplicates.
        self._read_lock = threading.Lock()
        self._read_locations: weakref.WeakValueDictionary[ObjectKey, MemoryObj] = (
            weakref.WeakValueDictionary()
        )
        self._event_bus = get_event_bus()
        self._eviction_controllers = []
        for manager, c in zip(self._l1_managers, self._l1_configs, strict=True):
            assert c.eviction is not None  # StorageManagerConfig validates this.
            self._eviction_controllers.append(L1EvictionController(manager, c.eviction))
        for eviction_controller in self._eviction_controllers:
            eviction_controller.start()

        # Multi-region transports use object pointers; single-region P2P/SHM
        # advertising remains available only for a single L1.
        self._l1_memory_desc = (
            self._l1_managers[0].get_l1_memory_desc()
            if len(self._l1_managers) == 1
            else None
        )
        self._next_adapter_id = 0
        # Serializes add_l2_adapter / delete_l2_adapter against each other.
        self._lifecycle_lock = threading.Lock()
        # Guards the _l2_adapters and _adapter_descriptors dicts.
        self._adapters_lock = threading.Lock()
        self._registered_l2_listeners: list[L2AdapterListener] = []
        self._l2_adapters: dict[int, L2AdapterInterface] = {}
        self._adapter_descriptors: dict[int, L2AdapterDescriptor] = {}
        for ac in config.l2_adapter_config.adapters:
            adapter_id, adapter, descriptor = self._build_l2_adapter(ac)
            self._l2_adapters[adapter_id] = adapter
            self._adapter_descriptors[adapter_id] = descriptor

        PeriodicEventNotifier.create(
            interval_ms=config.periodic_notifier_interval_ms,
            use_eventfd=HAS_EVENTFD,
        )

        # Per-cache_salt quota registry. Shared across the L2 eviction
        # controller (reads quotas each cycle) and the HTTP quota
        # endpoints (CRUD). Present even when no adapter uses
        # IsolatedLRU so the HTTP layer has a stable ``quota_manager``
        # reference. No explicit cleanup on close — the registry is
        # just a dict protected by a lock and has no OS resources.
        self._quota_manager = QuotaManager()

        # Unified L2 eviction controller for all adapters with eviction
        # config. Aggregate-usage policies (``LRU``, ``noop``) need
        # ``max_capacity_bytes > 0`` to compute a usage fraction;
        # adapters without capacity are skipped for those. Isolated
        # policies (``IsolatedLRU``) operate on per cache_salt byte
        # counts which the base class tracks regardless of capacity,
        # so they are wired up unconditionally.
        l2_eviction_states: list[L2AdapterEvictionState] = []
        for adapter_id, ac in zip(
            self._l2_adapters, config.l2_adapter_config.adapters, strict=True
        ):
            adapter = self._l2_adapters[adapter_id]
            if self._should_enable_l2_eviction(adapter, ac.eviction_config):
                assert ac.eviction_config is not None  # make linter happy
                l2_eviction_states.append(
                    L2AdapterEvictionState(
                        adapter_id=adapter_id,
                        adapter=adapter,
                        eviction_config=ac.eviction_config,
                    )
                )
        self._l2_eviction_controller = L2EvictionController(
            l2_eviction_states, quota_manager=self._quota_manager
        )
        self._l2_eviction_controller.start()

        # Controllers receive the initial set as ordered lists; they key
        # their own copies by ``descriptor.index`` (== adapter_id) and learn
        # of later changes via add_adapter/request_remove_adapter.
        # Each adapter's completion FD has exactly one store-controller owner.
        # Reuse the existing controller for each L1 and its affinity adapters.
        self._store_controllers = []
        for l1_config, manager in zip(self._l1_configs, self._l1_managers, strict=True):
            descriptors = [
                d
                for d in self._adapter_descriptors.values()
                if d.config.affinity_tag == l1_config.tag
            ]
            controller = StoreController(
                l1_manager=manager,
                l2_adapters=[self._l2_adapters[d.index] for d in descriptors],
                adapter_descriptors=descriptors,
                policy=create_store_policy(config.store_policy),
                l1_tag=l1_config.tag,
            )
            self._store_controllers.append(controller)
            controller.start()

        self._prefetch_controller = PrefetchController(
            l1_managers=self._l1_managers,
            l1_manager_descriptors=self._l1_descriptors,
            l2_adapters=list(self._l2_adapters.values()),
            adapter_descriptors=list(self._adapter_descriptors.values()),
            policy=create_prefetch_policy(config.prefetch_policy),
            max_in_flight=config.prefetch_max_in_flight,
            on_read_ready=self._record_read_locations,
        )
        self._prefetch_controller.start()

        # Compatibility for existing single-L1 integrations and diagnostics.
        if len(self._l1_managers) == 1:
            self._l1_manager = self._l1_managers[0]
            self._store_controller = self._store_controllers[0]
            self._eviction_controller = self._eviction_controllers[0]

        # L2 usage gauge — one observation per adapter, tagged by
        # ``l2_name``.  Parallel to L1Manager's ``l1_memory_usage_bytes``.
        register_gauge(
            "lmcache.l2",
            "lmcache_mp.l2_usage_bytes",
            (
                "Bytes currently held in each L2 adapter, tagged by "
                "``l2_name`` (one observation per adapter)."
            ),
            self.get_l2_usages,
        )

    # External APIs for serving engine integration code to call
    @enable_tracing()
    def reserve_write(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> dict[ObjectKey, MemoryObj]:
        """
        Reserve the object for writing into the storage manager.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for writing.
            layout_desc (MemoryLayoutDesc): Description of the memory layout
                for the objects to be reserved.
        Returns:
            dict[ObjectKey, MemoryObj]: A dictionary mapping object keys to their
                reserved memory objects. Note that not all requested keys could be
                reserved (e.g., out of memory or write conflict)
        """
        plan = self._write_policy.select_write_targets(keys, self._l1_descriptors)
        planned = [key for group in plan.values() for key in group]
        if (
            any(i < 0 or i >= len(self._l1_managers) for i in plan)
            or len(set(planned)) != len(planned)
            or not set(planned) <= set(keys)
        ):
            raise ValueError("Write policy must select at most one valid L1 per key")
        reserve_result: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.OUT_OF_MEMORY, None) for key in keys
        }
        with self._write_lock:
            routed: dict[int, list[ObjectKey]] = defaultdict(list)
            for index, group in plan.items():
                # Keep an in-flight write on its original manager if a policy
                # changes its choice between reserve_write and finish_write.
                for key in group:
                    location = self._write_locations.get(key)
                    routed[
                        self._object_owners[location] if location is not None else index
                    ].append(key)
            for index, available in routed.items():
                results = self._l1_managers[index].reserve_write(
                    keys=available,
                    is_temporary=[False] * len(available),
                    layout_desc=layout_desc,
                    tag=_L1_WRITE_TAG,
                )
                reserve_result.update(results)
                for key, (error, obj) in results.items():
                    if error == L1Error.SUCCESS and obj is not None:
                        self._object_owners[obj] = index
                        self._write_locations[key] = obj

        result = {k: m for k, (e, m) in reserve_result.items() if m is not None}
        successful_keys = list(result.keys())
        failed_keys = [k for k, (e, m) in reserve_result.items() if m is None]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_WRITE_RESERVED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

        oom_keys = [
            k for k, (e, _) in reserve_result.items() if e == L1Error.OUT_OF_MEMORY
        ]
        if oom_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={"during": "l1_store", "keys": oom_keys},
                )
            )

        return result

    @enable_tracing()
    def finish_write(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """
        Finish writing the objects into the storage manager.

        Admits the objects reserved by :meth:`reserve_write`: each becomes
        visible to readers unless the key is already resident, in which case
        the reserved copy is dropped.

        Args:
            keys (list[ObjectKey]): List of object keys that have been written.
        """
        finish_result = {}
        with self._write_lock:
            groups: dict[int, list[ObjectKey]] = defaultdict(list)
            for key in keys:
                location = self._write_locations.pop(key, None)
                if location is None:
                    finish_result[key] = L1Error.KEY_NOT_EXIST
                else:
                    groups[self._object_owners[location]].append(key)
            for index, group in groups.items():
                finish_result.update(
                    self._l1_managers[index].finish_write(group, tag=_L1_WRITE_TAG)
                )
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_WRITE_FINISHED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

        # TODO: global key states update

    @contextmanager
    def read_prefetched_results(
        self,
        keys: list[ObjectKey],
    ) -> Iterator[list[MemoryObj] | None]:
        """
        Read the memory objects from L1 storage that has been prefetched beforehand.
        Yielding an optional list of memory objects corresponding to the requested
        keys. If any the object is not found in L1, None is yielded.

        Args:
            keys (list[ObjectKey]): List of object keys to reserve for reading.

        Returns:
            Iterator[list[MemoryObj] | None]: An iterator yielding an optional list of
                memory objects corresponding to the requested keys.

        Note:
            If any object is not found in L1 storage, None is yielded. In this case,
            this function will release release the read lock of all successfully read
            memory objects when exiting the context.

            If the caller raised exception during the processing of the yielded memory
            objects, this function will ensure that the read locks will be decreased.
        """
        # Manual TRACE_CALL emission for the context manager.  The
        # ``@enable_tracing`` decorator cannot wrap a ``@contextmanager``
        # generator function (it would publish the call to the wrapper
        # rather than to ``__enter__``).  Emit enter/exit events
        # directly, gated on the tracing flag for zero overhead when
        # disabled.
        if is_tracing_enabled():
            publish_call_event(
                "lmcache.v1.distributed.storage_manager."
                "StorageManager.read_prefetched_results.__enter__",
                {"keys": keys},
            )
        read_results = self._read_objects(keys)
        good_keys: list[ObjectKey] = []
        good_objs: list[MemoryObj] = []
        bad_keys: list[ObjectKey] = []
        not_found_keys: list[ObjectKey] = []
        write_locked_keys: list[ObjectKey] = []
        all_good = True
        for k, (e, o) in read_results.items():
            if o is None:
                logger.error(
                    "Failed to read prefetched object %s from L1 storage: %s",
                    k,
                    strerror(e),
                )
                bad_keys.append(k)
                all_good = False
                if e == L1Error.KEY_NOT_EXIST:
                    not_found_keys.append(k)
                elif e == L1Error.KEY_NOT_READABLE:
                    write_locked_keys.append(k)
                continue

            good_keys.append(k)
            good_objs.append(o)

        # L1 read-failure anomaly reporting: unsafe_read is required to be
        # called post-reserve_read, so any failure here is a lock/eviction
        # race, not a normal cache miss.
        if not_found_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_READ_FAILED,
                    metadata={
                        "during": "l1_retrieve",
                        "reason": "not_found",
                        "keys": not_found_keys,
                    },
                )
            )
        if write_locked_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_READ_FAILED,
                    metadata={
                        "during": "l1_retrieve",
                        "reason": "write_locked",
                        "keys": write_locked_keys,
                    },
                )
            )

        successfully_yielded = False

        try:
            yield good_objs if all_good else None
            successfully_yielded = True
        except Exception:
            logger.exception(
                "Exception occurred while processing read prefetched results",
            )
            raise
        finally:
            # Decrease the read lock for all successfully read memory objects
            # if None is yielded or exception occurs during caller's processing
            if not all_good or not successfully_yielded:
                self._finish_read(good_keys)
                self._event_bus.publish(
                    Event(
                        event_type=EventType.SM_READ_PREFETCHED_FINISHED,
                        metadata={
                            "succeeded_keys": good_keys,
                            "failed_keys": bad_keys,
                        },
                    )
                )
            if is_tracing_enabled():
                publish_call_event(
                    "lmcache.v1.distributed.storage_manager."
                    "StorageManager.read_prefetched_results.__exit__",
                    {"keys": keys},
                )

    @enable_tracing()
    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
    ) -> None:
        """Finish reading prefetched objects.

        Args:
            keys: Object keys that have been read.
            read_locks: Read locks to release per key (the whole
                reservation when releasing a lookup's locks).
        """
        finish_result = self._finish_read(keys, read_locks)
        successful_keys = [k for k, e in finish_result.items() if e == L1Error.SUCCESS]
        failed_keys = [k for k, e in finish_result.items() if e != L1Error.SUCCESS]
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_READ_PREFETCHED_FINISHED,
                metadata={
                    "succeeded_keys": successful_keys,
                    "failed_keys": failed_keys,
                },
            )
        )

    @enable_tracing()
    def submit_prefetch_task(
        self,
        spec: PrefetchTaskSpec,
        external_request_id: str = "",
        skip_l2: bool = False,
    ) -> PrefetchHandle:
        """Prefetch objects into L1 asynchronously.

        Args:
            spec: The request (see :class:`PrefetchTaskSpec`).
            external_request_id: Caller id for end-to-end log tracing.
            skip_l2: If True, serve from L1 only. The result is available
                as soon as this returns.

        Returns:
            PrefetchHandle to track the task.
        """
        prefetch_request_id = self._prefetch_controller.submit_prefetch_request(
            spec, skip_l2=skip_l2
        )
        logger.debug(
            "Prefetch request submitted: %d keys in %d groups "
            "(external_request_id=%s, prefetch_request_id=%d, skip_l2=%s)",
            len(spec.key_groups) * spec.group_size,
            len(spec.key_groups),
            external_request_id,
            prefetch_request_id,
            skip_l2,
        )
        return PrefetchHandle(
            prefetch_request_id=prefetch_request_id,
            external_request_id=external_request_id,
            total_requested_keys=len(spec.key_groups) * spec.group_size,
            submit_time=time.monotonic(),
            sliding_windows=tuple(row.sliding_window_size for row in spec.key_groups),
        )

    def query_prefetch_status(self, handle: PrefetchHandle) -> PrefetchResult | None:
        """
        Query the status of the prefetch task.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.

        Returns:
            The task's result once it has finished, None while it is still
            in progress.

        Note:
            Each result is returned once; later calls for the same handle
            return None.
        """
        if handle.prefetch_request_id == -1:
            return PrefetchResult(hit_cells=[], l1_hit_cells=[], l2_hit_cells=[])
        result = self._prefetch_controller.query_prefetch_result(
            handle.prefetch_request_id
        )
        if result is None:
            return None
        total_hits = sum(row.popcount() for row in result.hit_cells)
        if total_hits > 0:
            elapsed_ms = (time.monotonic() - handle.submit_time) * 1000
            logger.info(
                "Prefetch request completed (L1+L2): "
                "%d/%d retained keys (%d L1, %d L2) in %.1f ms "
                "(external_request_id=%s, prefetch_request_id=%d)",
                total_hits,
                handle.total_requested_keys,
                result.l1_hit_count,
                result.l2_hit_count,
                elapsed_ms,
                handle.external_request_id,
                handle.prefetch_request_id,
            )
        return result

    @lmcache_deprecate(
        "the lookup hit is no longer reported before the prefetch finishes; "
        "use query_prefetch_status"
    )
    def query_prefetch_lookup_hits(
        self,
        handle: PrefetchHandle,
    ) -> int | None:
        """
        Query the number of prefix-hit chunks of a finished prefetch task.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.

        Returns:
            The number of prefix-hit chunks once the prefetch has finished,
            None while it is still in progress.

        Note:
            Consumes the result like ``query_prefetch_status``.
        """
        result = self.query_prefetch_status(handle)
        if result is None:
            return None
        if not result.hit_cells:
            return 0
        hit_length, _retain = fold_unfold_grouped(
            result.hit_cells, list(handle.sliding_windows)
        )
        return hit_length

    def wait_prefetch_status(
        self,
        handle: PrefetchHandle,
        timeout: float,
    ) -> bool:
        """
        Block until the prefetch task for ``handle`` has a result, or timeout.

        This lets a caller avoid busy-polling query_prefetch_status; the
        status itself is still retrieved via query_prefetch_status afterwards.

        Args:
            handle (PrefetchHandle): The handle of the prefetch task.
            timeout: Maximum number of seconds to wait for the result.

        Returns:
            True if a result is available within the timeout, False if the
            wait timed out.
        """
        if handle.prefetch_request_id == -1:
            return True
        return self._prefetch_controller.wait_prefetch_result(
            handle.prefetch_request_id, timeout
        )

    def touch_l1_keys(self, keys: list[ObjectKey]):
        """
        Touch the keys in L1 storage, marking the keys
        as accessed(retrieved or stored).

        Args:
            keys (list[ObjectKey]): List of object keys to touch.
        """
        for manager in self._l1_managers:
            manager.touch_keys(keys)

    def delete_l1_keys(
        self, keys: list[ObjectKey], force: bool = False
    ) -> tuple[int, int]:
        """Delete the given keys from L1.

        Args:
            keys (list[ObjectKey]): List of object keys to delete.
            force (bool): When True, delete even read/write-locked keys; else
                skip them.

        Returns:
            tuple[int, int]: ``(deleted, skipped)`` -- the number of keys removed
                and the number refused because they were locked (non-force only).
                Missing keys are a no-op, so the operation is idempotent.
        """
        results = {k: L1Error.KEY_NOT_EXIST for k in keys}
        for manager in self._l1_managers:
            for key, error in manager.delete(keys, force=force).items():
                if error != L1Error.KEY_NOT_EXIST:
                    results[key] = error
        deleted = sum(1 for err in results.values() if err == L1Error.SUCCESS)
        skipped = sum(1 for err in results.values() if err == L1Error.KEY_IS_LOCKED)
        return deleted, skipped

    def unsafe_read(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[MemoryObj]]:
        """Read already read-locked objects without acquiring new read locks."""
        read_results = self._read_objects(keys)
        good_keys: list[ObjectKey] = []
        good_objs: list[MemoryObj] = []
        for key in keys:
            err, obj = read_results.get(key, (L1Error.KEY_NOT_EXIST, None))
            if err != L1Error.SUCCESS or obj is None:
                continue
            good_keys.append(key)
            good_objs.append(obj)
        return good_keys, good_objs

    @property
    def quota_manager(self) -> QuotaManager:
        """Per-cache_salt quota registry.

        Exposed so the HTTP layer can serve CRUD endpoints without
        reaching into private state. Always non-``None`` — the
        storage manager creates the registry at construction time.
        """
        return self._quota_manager

    @property
    def l1_memory_desc(self) -> L1MemoryDesc:
        """Return the single L1 buffer descriptor.

        Raises:
            ValueError: If L1 does not expose exactly one registerable region.
        """
        if self._l1_memory_desc is None:
            raise ValueError("L1 does not expose a single registerable memory region")
        return self._l1_memory_desc

    def get_l2_usages(
        self,
    ) -> list[tuple[int | float, dict[str, object]]]:
        """Per-adapter L2 usage in OTel-observation shape.

        Backing data for the ``lmcache_mp.l2_usage_bytes`` observable
        gauge.  One entry per configured adapter.

        Returns:
            A list of ``(total_bytes_used, {"l2_name": <type_name>})``
            tuples — empty when no L2 adapters are configured.  Adapters
            whose ``get_usage()`` raises are skipped (the gauge prefers
            silence over a poison observation).
        """
        return [
            (int(usage.total_bytes_used), {"l2_name": type_name})
            for type_name, usages in self.get_l2_usages_by_type().items()
            for usage in usages
        ]

    def get_l2_usages_by_type(self) -> dict[str, list[AdapterUsage]]:
        """Per-adapter usage snapshots grouped by adapter type name.

        Backing data for usage telemetry's per-connector presence and
        occupancy reporting and for the :meth:`get_l2_usages` gauge.
        Adapters whose ``get_usage()`` raises are skipped.

        Returns:
            Mapping from adapter type name (e.g. ``"dax"``) to one usage
            snapshot per active adapter of that type; empty when no L2
            adapters are configured.
        """
        out_by_type: dict[str, list[AdapterUsage]] = {}
        for _adapter_id, desc, adapter in self._snapshot_adapters():
            try:
                usage = adapter.get_usage()
            except Exception:
                logger.exception(
                    "L2 adapter %s get_usage() failed; skipping in usage snapshot",
                    desc.type_name,
                )
                continue
            out_by_type.setdefault(desc.type_name, []).append(usage)
        return out_by_type

    def publish_capacity(self) -> None:
        """Announce the current capacity topology on the event bus.

        Called after every coordinator registration, so a restarted
        coordinator relearns this server's capacities even if nothing is
        ever reconfigured. Later changes announce themselves.
        """
        self._publish_capacity_changed()

    def _build_capacities(self) -> list[ModuleMemoryCapacity]:
        """Assemble one capacity entry per memory compartment.

        Returns:
            L1 per backing medium, then one entry per L2 adapter.
        """
        l1_capacity: dict[L1BackendType, int] = defaultdict(int)
        for config in self._l1_configs:
            for backend, configured in get_configured_capacity_bytes(config).items():
                l1_capacity[backend] += configured
        capacities = [
            ModuleMemoryCapacity(
                tier=Tier.L1,
                backend=backend.value,
                capacity_bytes=configured,
                shared=False,
            )
            for backend, configured in l1_capacity.items()
        ]
        for _adapter_id, desc, adapter in self._snapshot_adapters():
            try:
                usage = adapter.get_usage()
            except Exception:
                logger.exception(
                    "L2 adapter %s get_usage() failed; omitting from the "
                    "capacity report",
                    desc.type_name,
                )
                continue
            capacities.append(
                ModuleMemoryCapacity(
                    tier=Tier.L2,
                    backend=desc.type_name,
                    capacity_bytes=int(usage.total_capacity_bytes),
                    shared=bool(desc.config.shared),
                )
            )
        return capacities

    def get_l1_usage(self) -> tuple[int, int]:
        """Current occupancy of the L1 memory pool.

        Backing data for usage telemetry's L1 occupancy reporting.

        Returns:
            Tuple of ``(used_bytes, total_bytes)``.
        """
        usages = [m.get_memory_usage() for m in self._l1_managers]
        return sum(u for u, _ in usages), sum(t for _, t in usages)

    def get_usage_bytes_by_cache_salt(self) -> dict[str, int]:
        """Aggregate ``cache_salt`` byte usage across every L2 adapter.

        Used by the HTTP quota endpoints to report ``current_usage_gb``
        alongside the configured limit. Aggregation is a simple sum:
        each adapter tracks the same salt independently so the totals
        are additive.
        """
        totals: dict[str, int] = {}
        for _adapter_id, _desc, adapter in self._snapshot_adapters():
            snap = adapter.get_usage().bytes_by_cache_salt
            for salt, used in snap.items():
                totals[salt] = totals.get(salt, 0) + used
        return totals

    # L2 APIs
    def get_l2_adapter_reconfigure_status(self) -> dict:
        """Return status for all runtime-reconfigurable L2 adapters.

        Returns:
            JSON-serializable status. If no reconfigurable adapter is configured,
            ``enabled`` is ``False`` and the adapter list is empty.
        """
        type_names = {
            adapter_id: desc.type_name
            for adapter_id, desc, _ in self._snapshot_adapters()
        }
        adapters = []
        for adapter_index, (
            l2_adapter_index,
            adapter,
        ) in enumerate(self._list_reconfigurable_l2_adapters()):
            status = dict(adapter.reconfigure_status())
            if l2_adapter_index in type_names:
                status["backend"] = type_names[l2_adapter_index]
            status["adapter_index"] = adapter_index
            status["l2_adapter_index"] = l2_adapter_index
            adapters.append(status)

        return {
            "enabled": bool(adapters),
            "num_adapters": len(adapters),
            "adapters": adapters,
        }

    def reconfigurable_l2_backends(self) -> set[str]:
        """Return the ``type_name`` of every L2 adapter that supports runtime
        reconfiguration.

        Returns:
            The set of reconfigurable adapter ``type_name`` strings (empty when
            none are reconfigurable). The ``{backend}`` path parameter the
            ``/reconfigure`` routes expect is the adapter's ``type_name``.
        """
        return {
            desc.type_name
            for _adapter_id, desc, adapter in self._snapshot_adapters()
            if self._unwrap_reconfigurable_l2_adapter(adapter) is not None
        }

    def _publish_capacity_changed(self) -> None:
        """Announce the current capacity topology on the event bus.

        Lock-free. ``_build_capacities`` guards its own reads
        (``_snapshot_adapters`` takes ``_adapters_lock``), and ordering is
        not this class's problem: the cache-event subscriber numbers
        declarations as it emits them, on the one bus drain thread, so a
        number cannot come apart from the topology it labels. Callers here
        are concurrent -- registration publishes from the event loop while a
        worker may be adding an adapter -- which is exactly why the counter
        does not live here.

        The event carries the whole topology, not a delta, so a dropped one
        is repaired by the next rather than leaving the coordinator
        permanently wrong.
        """
        self._event_bus.publish(
            Event(
                event_type=EventType.SM_CAPACITY_CHANGED,
                metadata={
                    "snapshot": CapacitySnapshot(
                        modules=tuple(self._build_capacities())
                    )
                },
            )
        )

    def reconfigure_l2_adapter(
        self,
        adapter_index: int,
        operation: str,
        payload: dict[str, object],
    ) -> dict:
        """Route a runtime reconfiguration request to one L2 adapter.

        Args:
            adapter_index: Zero-based reconfigurable-adapter index.
            operation: Adapter-specific operation name.
            payload: Adapter-specific operation payload.

        Returns:
            JSON-serializable operation result.
        """
        adapter = self._get_reconfigurable_l2_adapter(adapter_index)
        result = adapter.reconfigure(operation, payload)
        result["adapter_index"] = adapter_index
        # Lock-free: reconfigure did not serialize against adapter
        # add/delete before, and publishing is no reason to start.
        self._publish_capacity_changed()
        return result

    def add_l2_adapter(self, config: L2AdapterConfigBase) -> int:
        """Blocking function to add a new L2 adapter at runtime. Thread-safe.

        Args:
            config: The adapter configuration.

        Returns:
            The stable id assigned to the new adapter.
        """
        with self._lifecycle_lock:
            adapter_id, adapter, descriptor = self._build_l2_adapter(config)
            for listener in self._registered_l2_listeners:
                adapter.register_listener(listener)
            with self._adapters_lock:
                self._l2_adapters[adapter_id] = adapter
                self._adapter_descriptors[adapter_id] = descriptor
            self._store_controllers[self._l1_by_tag[config.affinity_tag]].add_adapter(
                adapter_id, adapter, descriptor
            )
            self._prefetch_controller.add_adapter(adapter_id, adapter, descriptor)
            if self._should_enable_l2_eviction(adapter, config.eviction_config):
                assert config.eviction_config is not None  # make linter happy
                self._l2_eviction_controller.add_adapter_state(
                    L2AdapterEvictionState(
                        adapter_id=adapter_id,
                        adapter=adapter,
                        eviction_config=config.eviction_config,
                    )
                )
            logger.info("Added L2 adapter %d (%s)", adapter_id, descriptor.type_name)
            self._publish_capacity_changed()
            return adapter_id

    def delete_l2_adapter(self, adapter_id: int, timeout: float = 30.0) -> None:
        """Blocking function to drain the L2 adapter gracefully at runtime.
        Thread-safe.

        Stops routing new stores/prefetches to the adapter, waits for its
        in-flight work to finish, removes it from the controllers, and
        closes it.

        Args:
            adapter_id: Stable id of the adapter to remove.
            timeout: Maximum seconds to wait for in-flight work to drain.

        Raises:
            ValueError: If no adapter with ``adapter_id`` is active.
            TimeoutError: If draining did not complete within ``timeout``;
                the adapter is left active (draining) so the caller can
                retry.
        """
        with self._lifecycle_lock:
            if adapter_id not in self._l2_adapters:
                raise ValueError(f"No L2 adapter with id {adapter_id}")

            deadline = time.monotonic() + timeout
            affinity = self._adapter_descriptors[adapter_id].config.affinity_tag
            store_done = self._store_controllers[
                self._l1_by_tag[affinity]
            ].request_remove_adapter(adapter_id)
            prefetch_done = self._prefetch_controller.request_remove_adapter(adapter_id)
            if not store_done.wait(timeout=max(0.0, deadline - time.monotonic())):
                raise LMCacheTimeoutError(
                    f"Timed out draining adapter {adapter_id} from store controller"
                )
            if not prefetch_done.wait(timeout=max(0.0, deadline - time.monotonic())):
                raise LMCacheTimeoutError(
                    f"Timed out draining adapter {adapter_id} from prefetch controller"
                )

            self._l2_eviction_controller.remove_adapter_state(adapter_id)
            with self._adapters_lock:
                adapter = self._l2_adapters.pop(adapter_id)
                self._adapter_descriptors.pop(adapter_id, None)
            adapter.close()
            logger.info("Deleted L2 adapter %d", adapter_id)
            self._publish_capacity_changed()

    def l2_adapters(self) -> list[tuple[L2AdapterDescriptor, L2AdapterInterface]]:
        """Return all active L2 adapters paired with descriptors, in
        ascending adapter-id order (== configuration order for the initial
        set, then runtime-added adapters). The list is empty when no L2 is
        configured.

        Do not cache the returned pairs — ``reconfigure_l2_adapter``,
        ``add_l2_adapter``, and ``delete_l2_adapter`` may change the set at
        runtime.
        """
        return [
            (desc, adapter) for _adapter_id, desc, adapter in self._snapshot_adapters()
        ]

    # Management APIs
    def clear(self, force: bool = False):
        """
        Clear data in the storage manager.

        Args:
            force: If True, clear ALL objects including locked ones.
                This may corrupt in-flight store/prefetch operations.
                If False (default), only clear unlocked objects, keeping
                write-locked and read-locked objects intact.
        """
        for manager in self._l1_managers:
            manager.clear(force=force)

    def close(self):
        """
        Close the storage manager and release all resources.
        """
        self._prefetch_controller.stop()
        for controller in [*self._store_controllers, *self._eviction_controllers]:
            controller.stop()
        self._l2_eviction_controller.stop()

        PeriodicEventNotifier.shutdown()

        for adapter in self._l2_adapters.values():
            adapter.close()

        for manager in self._l1_managers:
            manager.close()

    def report_status(self) -> dict:
        """Return a status dict aggregating all sub-component statuses."""
        l1s = {
            c.tag: m.report_status()
            for c, m in zip(self._l1_configs, self._l1_managers, strict=True)
        }
        stores = {
            c.tag: m.report_status()
            for c, m in zip(self._l1_configs, self._store_controllers, strict=True)
        }
        evictions = {
            c.tag: m.report_status()
            for c, m in zip(self._l1_configs, self._eviction_controllers, strict=True)
        }
        prefetch = self._prefetch_controller.report_status()
        l2_eviction = self._l2_eviction_controller.report_status()
        adapters = [a.report_status() for _id, _desc, a in self._snapshot_adapters()]
        children = [
            *l1s.values(),
            *stores.values(),
            *evictions.values(),
            prefetch,
            l2_eviction,
            *adapters,
        ]
        result = {
            "is_healthy": all(c["is_healthy"] for c in children),
            "l1_managers": l1s,
            "store_controllers": stores,
            "l1_eviction_controllers": evictions,
            "prefetch_controller": prefetch,
            "l2_eviction_controller": l2_eviction,
            "l2_adapters": adapters,
            "num_l2_adapters": len(adapters),
        }
        if len(l1s) == 1:
            result.update(
                l1_manager=next(iter(l1s.values())),
                store_controller=next(iter(stores.values())),
                l1_eviction_controller=next(iter(evictions.values())),
            )
        return result

    def register_l2_listener(self, listener: L2AdapterListener) -> None:
        """Register a listener on all current and future L2 adapters.

        The listener is recorded so that adapters added later via
        :meth:`add_l2_adapter` receive it too.

        Args:
            listener: The listener to register.
        """
        with self._lifecycle_lock:
            self._registered_l2_listeners.append(listener)
            for adapter in self._l2_adapters.values():
                adapter.register_listener(listener)

    # Functions for debugging and testing
    def memcheck(self) -> bool:
        """
        Perform memory check for all storage tiers.

        Returns:
            True if memory is consistent, False otherwise.
        """
        return all([m.memcheck() for m in self._l1_managers])

    def _record_read_locations(
        self, groups: dict[int, list[ObjectKey]], read_locks: int
    ) -> None:
        """Keep overlapping prefetches on one locked copy per key."""
        with self._read_lock:
            for index, keys in groups.items():
                manager = self._l1_managers[index]
                for key, (error, obj) in manager.unsafe_read(keys).items():
                    if error != L1Error.SUCCESS or obj is None:
                        continue
                    location = self._read_locations.get(key)
                    owner = index
                    if location is not None and self._object_owners[location] != index:
                        previous = self._l1_managers[self._object_owners[location]]
                        err, existing = previous.reserve_read([key], read_locks)[key]
                        if err == L1Error.SUCCESS and existing is not None:
                            manager.finish_read([key], read_locks)
                            previous.touch_keys([key])
                            owner, obj = self._object_owners[location], existing
                    self._object_owners[obj] = owner
                    self._read_locations[key] = obj

    def _read_objects(
        self, keys: list[ObjectKey]
    ) -> dict[ObjectKey, L1OperationResult]:
        with self._read_lock:
            results: dict[ObjectKey, L1OperationResult] = {
                key: (L1Error.KEY_NOT_EXIST, None) for key in keys
            }
            groups: dict[int, list[ObjectKey]] = defaultdict(list)
            for key in keys:
                location = self._read_locations.get(key)
                if location is not None:
                    groups[self._object_owners[location]].append(key)
                elif len(self._l1_managers) == 1:
                    groups[0].append(key)
            for index, group in groups.items():
                results.update(self._l1_managers[index].unsafe_read(group))
            return results

    def _finish_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1Error]:
        with self._read_lock:
            results = {key: L1Error.KEY_NOT_EXIST for key in keys}
            groups: dict[int, list[ObjectKey]] = defaultdict(list)
            for key in keys:
                location = self._read_locations.get(key)
                if location is not None:
                    groups[self._object_owners[location]].append(key)
                elif len(self._l1_managers) == 1:
                    groups[0].append(key)
            for index, group in groups.items():
                results.update(self._l1_managers[index].finish_read(group, read_locks))
            return results

    def _snapshot_adapters(
        self,
    ) -> list[tuple[int, L2AdapterDescriptor, L2AdapterInterface]]:
        """Snapshot the active adapters under the lock, in ascending
        adapter-id order. Iterate this instead of the live dicts so a
        concurrent add/delete cannot change them mid-iteration.

        Returns:
            A list of ``(adapter_id, descriptor, adapter)`` tuples.
        """
        with self._adapters_lock:
            return [
                (adapter_id, self._adapter_descriptors[adapter_id], adapter)
                for adapter_id, adapter in sorted(self._l2_adapters.items())
            ]

    def _has_l2_adapters(self) -> bool:
        """Return whether any L2 adapter is currently active."""
        with self._adapters_lock:
            return bool(self._l2_adapters)

    def _build_l2_adapter(
        self,
        config: L2AdapterConfigBase,
    ) -> tuple[int, L2AdapterInterface, L2AdapterDescriptor]:
        """Create a L2 adapter instance based on the config.

        Args:
            config: The adapter configuration.

        Returns:
            A ``(adapter_id, adapter, descriptor)`` tuple. ``adapter_id`` is
            the freshly allocated stable id, ``adapter`` is the new adapter
            instance, and ``descriptor`` is its descriptor carrying that id.
        """
        adapter_id = self._next_adapter_id
        self._next_adapter_id += 1
        if config.affinity_tag not in self._l1_by_tag:
            raise ValueError(f"Unknown L1 affinity_tag: {config.affinity_tag!r}")
        manager = self._l1_managers[self._l1_by_tag[config.affinity_tag]]
        adapter: L2AdapterInterface = create_l2_adapter(
            config, manager.get_l1_memory_desc()
        )
        if config.serde_config is not None:
            adapter = SerdeL2AdapterWrapper(
                inner=adapter,
                serde=create_serde_processor(config.serde_config),
                l1_manager=manager,
            )
        descriptor = L2AdapterDescriptor(index=adapter_id, config=config)
        # Stamp the registered type name so the adapter's cache events on
        # the observability bus carry their backend identity.
        adapter.set_backend_identity(descriptor.type_name, shared=config.shared)
        return adapter_id, adapter, descriptor

    def _should_enable_l2_eviction(
        self,
        adapter: L2AdapterInterface,
        eviction_config: EvictionConfig | None,
    ) -> bool:
        """Whether to wire an adapter into the L2 eviction controller.

        Args:
            adapter: The adapter to evaluate.
            eviction_config: The adapter's eviction config, if any.

        Returns:
            True if an eviction state should be created for this adapter.
        """
        if eviction_config is None:
            return False
        policy_name = eviction_config.eviction_policy
        if policy_name != "IsolatedLRU" and not adapter.supports_global_eviction:
            logger.warning(
                "L2 adapter %s configured with '%s' eviction but does "
                "not support global eviction (max_capacity_bytes=0); "
                "skipping aggregate-usage eviction setup.",
                type(adapter).__name__,
                policy_name,
            )
            return False
        return True

    def _unwrap_reconfigurable_l2_adapter(
        self,
        adapter: L2AdapterInterface,
    ) -> Optional[L2ReconfigurableAdapter]:
        if isinstance(adapter, L2ReconfigurableAdapter):
            return adapter

        inner = getattr(adapter, "inner_adapter", None)
        if inner is not None and isinstance(inner, L2ReconfigurableAdapter):
            return inner

        return None

    def _list_reconfigurable_l2_adapters(
        self,
    ) -> list[tuple[int, L2ReconfigurableAdapter]]:
        with self._adapters_lock:
            items = sorted(self._l2_adapters.items())
        adapters: list[tuple[int, L2ReconfigurableAdapter]] = []
        for l2_adapter_index, adapter in items:
            reconfigurable_adapter = self._unwrap_reconfigurable_l2_adapter(adapter)
            if reconfigurable_adapter is not None:
                adapters.append((l2_adapter_index, reconfigurable_adapter))
        return adapters

    def _get_reconfigurable_l2_adapter(
        self,
        adapter_index: int,
    ) -> L2ReconfigurableAdapter:
        adapters = self._list_reconfigurable_l2_adapters()
        if adapter_index < 0 or adapter_index >= len(adapters):
            raise L2ReconfigureError(404, "L2 adapter not reconfigurable")
        return adapters[adapter_index][1]
