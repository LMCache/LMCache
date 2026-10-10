# SPDX-License-Identifier: Apache-2.0
"""
Distributed multi-tier storage manager for MP mode
"""

# Standard
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from typing import Any, Iterator, Optional, cast
import threading
import time

# First Party
from lmcache.lmcache_native import PeriodicEventNotifier
from lmcache.logging import init_logger
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
from lmcache.v1.distributed.config import (
    EvictionConfig,
    StorageManagerConfig,
    requires_single_l1_memory_region,
    unwrap_l2_adapter_config,
)
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError, strerror
from lmcache.v1.distributed.internal_api import (
    L1ManagerInterface,
    L1MemoryDesc,
    L1OperationResult,
    L2AdapterListener,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import AdapterUsage, L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.reconfiguration import (
    L2DeviceOwner,
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
from lmcache.v1.distributed.storage_controllers.write_policy import OrderedWritePolicy
from lmcache.v1.memory_allocators.devdax_memory_allocator import (
    DevDaxArenaState,
    DevDaxArenaStatus,
    DevDaxRemoveMode,
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

# Internal stream-callback payload. No MemoryObj or device pointer crosses msgpack.
L1WriteCompletion = list[tuple[int, list[ObjectKey]]]


class StorageManager:
    def __init__(
        self,
        config: StorageManagerConfig,
        *,
        _l1_managers: tuple[L1ManagerInterface, ...] | None = None,
        _write_policy: OrderedWritePolicy | None = None,
    ) -> None:
        """Create configured peer L1s and their affinity-bound L2 controllers.

        Args:
            config: Backend, capacity, eviction, and fixed-affinity settings.
            _l1_managers: Internal test injection of existing ordered managers.
                Validated managers transfer lifecycle ownership to this instance.
            _write_policy: Candidate order, validated and captured once at startup.
                Later policy mutations do not reconfigure the manager.

        Raises:
            ValueError: Manager identities/tags, placement, backing storage, or
                affinity settings conflict.
        """
        with ExitStack() as cleanup:
            if _l1_managers is not None:
                if not _l1_managers or len(
                    {m.l1_manager_id for m in _l1_managers}
                ) != len(_l1_managers):
                    raise ValueError("L1 managers must be nonempty and distinct")
                managers = _l1_managers
                config = replace(
                    config, l1_manager_configs=[m.config for m in managers]
                )
            else:
                created: list[L1ManagerInterface] = []
                for manager_config in config.l1_manager_configs:
                    path = manager_config.memory_config.devdax_path
                    if path and any(m.owns_device(path) for m in created):
                        raise ValueError(
                            f"Device-DAX path already owned by an L1: {path}"
                        )
                    manager: L1ManagerInterface = L1Manager(manager_config)
                    cleanup.callback(manager.close)
                    created.append(manager)
                managers = tuple(created)
            self._l1_configs = (
                [m.config for m in managers]
                if _l1_managers is not None
                else config.l1_manager_configs
            )
            self._l1_by_tag = {
                c.tag: m for c, m in zip(self._l1_configs, managers, strict=True)
            }
            self._l1_manager = managers[0]
            self._l1_managers_by_id = {m.l1_manager_id: m for m in managers}
            policy = _write_policy or OrderedWritePolicy(tuple(self._l1_managers_by_id))
            candidates = policy.select_write_targets()
            if len(set(candidates)) != len(candidates) or any(
                owner not in self._l1_managers_by_id for owner in candidates
            ):
                raise ValueError(
                    "write policy must select distinct registered L1 managers"
                )
            # Topology is fixed; resolve the validated order outside the write path.
            self._write_managers = tuple(
                self._l1_managers_by_id[owner] for owner in candidates
            )
            if _l1_managers is not None:
                for manager in managers:
                    cleanup.callback(manager.close)
            self._event_bus = get_event_bus()

            self._eviction_controllers = [
                L1EvictionController(m, c.eviction or config.eviction_config)
                for c, m in zip(self._l1_configs, managers, strict=True)
            ]
            for eviction_controller in self._eviction_controllers:
                eviction_controller.start()
                cleanup.callback(eviction_controller.stop)
            self._eviction_controller = self._eviction_controllers[0]
            self._l1_memory_desc = self._l1_manager.get_l1_memory_desc()
            self._next_adapter_id = 0
            # Serializes L1/L2 additions and L2 adapter registration/deletion.
            # Held from ownership check through add, before allocator/device locks.
            self._lifecycle_lock = threading.Lock()
            # Keeps capacity snapshots ordered by the point at which they are
            # built. Registration can publish concurrently with runtime changes.
            self._capacity_publish_lock = threading.Lock()
            # Guards the _l2_adapters and _adapter_descriptors dicts.
            self._adapters_lock = threading.Lock()
            self._registered_l2_listeners: list[L2AdapterListener] = []
            self._l2_adapters: dict[int, L2AdapterInterface] = {}
            self._adapter_descriptors: dict[int, L2AdapterDescriptor] = {}
            for ac in config.l2_adapter_config.adapters:
                adapter_id, adapter, descriptor = self._build_l2_adapter(ac)
                cleanup.callback(adapter.close)
                self._l2_adapters[adapter_id] = adapter
                self._adapter_descriptors[adapter_id] = descriptor

            PeriodicEventNotifier.create(
                interval_ms=config.periodic_notifier_interval_ms,
                use_eventfd=HAS_EVENTFD,
            )

            cleanup.callback(PeriodicEventNotifier.shutdown)

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
            cleanup.callback(self._l2_eviction_controller.stop)

            # Controllers receive the initial set as ordered lists; they key
            # their own copies by ``descriptor.index`` (== adapter_id) and learn
            # of later changes via add_adapter/request_remove_adapter.
            self._store_controllers: dict[int, StoreController] = {}
            for c, manager in zip(self._l1_configs, managers, strict=True):
                descriptors = [
                    d
                    for d in self._adapter_descriptors.values()
                    if d.config.affinity_tag == c.tag
                ]
                controller = StoreController(
                    manager,
                    [self._l2_adapters[d.index] for d in descriptors],
                    descriptors,
                    create_store_policy(config.store_policy),
                )
                self._store_controllers[manager.l1_manager_id] = controller
                controller.start()
                cleanup.callback(controller.stop)
            self._store_controller = self._store_controllers[managers[0].l1_manager_id]
            self._prefetch_controller = PrefetchController(
                l1_managers=list(managers),
                l1_manager_descriptors=[
                    L1ManagerDescriptor(index=m.l1_manager_id, config=c)
                    for m, c in zip(managers, self._l1_configs, strict=True)
                ],
                l2_adapters=list(self._l2_adapters.values()),
                adapter_descriptors=list(self._adapter_descriptors.values()),
                policy=create_prefetch_policy(config.prefetch_policy),
                max_in_flight=config.prefetch_max_in_flight,
            )
            self._prefetch_controller.start()
            cleanup.callback(self._prefetch_controller.stop)

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
            cleanup.pop_all()

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

        Only OUT_OF_MEMORY keys advance to the next candidate. Each L1 receives
        the whole pending subset; overflow does not split allocation batches.
        Candidate managers and their order are captured at construction.

        Raises:
            Exception: Allocation exceptions propagate; they are not overflow.
        """
        reserve_result = self._reserve_write_with_status(keys, layout_desc)
        return {k: m for k, (_, m) in reserve_result.items() if m is not None}

    @enable_tracing()
    def reserve_write_with_status(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> dict[ObjectKey, tuple[L1Error, MemoryObj | None]]:
        """Reserve objects for writing and preserve each key's status.

        Unlike :meth:`reserve_write`, this method does not discard failed
        reservation results. Callers that require fail-closed behavior can
        therefore distinguish an allocation failure from a skipped key.

        Args:
            keys: Object keys to reserve for writing.
            layout_desc: Memory layout of each requested object.

        Returns:
            A mapping from every requested key to its L1 error and optional
            reserved memory object.
        """
        return self._reserve_write_with_status(keys, layout_desc)

    @enable_tracing()
    def abort_write(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1Error]:
        """Discard staged writes without publishing them to readers.

        Args:
            keys: Keys successfully returned by a preceding write reservation.

        Returns:
            A mapping from every requested key to its L1 completion status.

        Raises:
            ValueError: Multiple managers require captured owner/key groups via
                :meth:`prepare_write_completion` and
                :meth:`abort_write_by_owner` instead.
        """
        self._require_single_l1()
        return self._l1_manager.finish_write_and_delete(keys, tag=_L1_WRITE_TAG)

    def abort_write_by_owner(self, completion: L1WriteCompletion) -> None:
        """Discard captured reservations without publishing them to readers.

        Args:
            completion: Owner/key groups captured by
                :meth:`prepare_write_completion`.

        Raises:
            ValueError: A captured owner is no longer registered.
        """
        if any(owner not in self._l1_managers_by_id for owner, _ in completion):
            raise ValueError("write completion requires a registered L1 owner")
        for owner, keys in completion:
            self._l1_managers_by_id[owner].finish_write_and_delete(
                keys, tag=_L1_WRITE_TAG
            )

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

        Raises:
            ValueError: Multiple managers require captured owner/key groups via
                prepare_write_completion and finish_write_by_owner instead.
        """
        self._require_single_l1()
        self.finish_write_by_owner([(self._l1_manager.l1_manager_id, keys)])

    def prepare_write_completion(
        self, objects: dict[ObjectKey, MemoryObj]
    ) -> L1WriteCompletion:
        """Capture owner/key groups from the actual reserved objects.

        Args:
            objects: Original key/object pairs returned by reserve_write.

        Returns:
            A MessagePack-compatible batch for finish_write_by_owner.

        Raises:
            ValueError: An object has no owner or belongs to another manager set.
        """
        groups: dict[int, list[ObjectKey]] = {}
        for key, obj in objects.items():
            owner = obj.get_l1_manager()
            if owner is None or owner not in self._l1_managers_by_id:
                raise ValueError("write completion requires a registered L1 owner")
            groups.setdefault(owner, []).append(key)
        return list(groups.items())

    def finish_write_by_owner(self, completion: L1WriteCompletion) -> None:
        """Finish captured reservations without rerunning placement policy.

        Args:
            completion: Owner/key groups captured before scheduling completion.

        Raises:
            ValueError: A captured owner is no longer registered.

        The caller must wait for device writes. L1 staging, writer tags, and TTLs
        retain their existing lifetime rules; an owner tag is not a write epoch.
        Single-L1 tracing retains the replayable finish_write(keys) record.
        """
        if any(owner not in self._l1_managers_by_id for owner, _ in completion):
            raise ValueError("write completion requires a registered L1 owner")
        if is_tracing_enabled() and len(self._l1_managers_by_id) == 1:
            # Replay creates fresh manager IDs; keep its existing key-only schema
            # and emit once for both the legacy and owner-routed entry points.
            publish_call_event(
                "lmcache.v1.distributed.storage_manager.StorageManager.finish_write",
                {"keys": [key for _, keys in completion for key in keys]},
            )
        finish_result: dict[ObjectKey, L1Error] = {}
        for owner, keys in completion:
            finish_result.update(
                self._l1_managers_by_id[owner].finish_write(keys, tag=_L1_WRITE_TAG)
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
        l1_owners: dict[ObjectKey, int] | None = None,
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
        read_results = self._read_objects(keys, l1_owners)
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
                self._finish_read_objects(good_keys, 1, l1_owners)
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

    @enable_tracing(redact=("l1_owners",))
    def finish_read_prefetched(
        self,
        keys: list[ObjectKey],
        read_locks: int = 1,
        l1_owners: dict[ObjectKey, int] | None = None,
    ) -> None:
        """Finish reading prefetched objects.

        Args:
            keys: Object keys that have been read.
            read_locks: Read locks to release per key (the whole
                reservation when releasing a lookup's locks).
        """
        finish_result = self._finish_read_objects(keys, read_locks, l1_owners)
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
        for manager in self._l1_managers_by_id.values():
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
                A key is skipped if any L1 copy remains locked.
                Missing keys are a no-op, so the operation is idempotent.
        """
        results: dict[ObjectKey, L1Error] = {}
        for manager in self._l1_managers_by_id.values():
            for key, error in manager.delete(keys, force=force).items():
                if error == L1Error.KEY_IS_LOCKED or (
                    error != L1Error.KEY_NOT_EXIST
                    and results.get(key) != L1Error.KEY_IS_LOCKED
                ):
                    results[key] = error
        deleted = sum(1 for err in results.values() if err == L1Error.SUCCESS)
        skipped = sum(1 for err in results.values() if err == L1Error.KEY_IS_LOCKED)
        return deleted, skipped

    def unsafe_read(
        self,
        keys: list[ObjectKey],
        l1_owners: dict[ObjectKey, int] | None = None,
    ) -> tuple[list[ObjectKey], list[MemoryObj]]:
        """Read already read-locked objects without acquiring new read locks."""
        read_results = self._read_objects(keys, l1_owners)
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
        """Descriptor of the L1 memory buffer backing this storage manager.

        Raises:
            ValueError: More than one L1 is configured, or the L1 has no
                registerable buffer (GDS).
        """
        self._require_single_l1()
        if self._l1_memory_desc is None:
            raise ValueError("The L1 exposes no registerable memory buffer")
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

    def _reserve_write_with_status(
        self,
        keys: list[ObjectKey],
        layout_desc: MemoryLayoutDesc,
    ) -> dict[ObjectKey, tuple[L1Error, MemoryObj | None]]:
        """Reserve writes and publish reservation outcome events."""
        reserve_result: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.OUT_OF_MEMORY, None) for key in keys
        }
        pending = keys
        for manager in self._write_managers:
            if not pending:
                break
            results = manager.reserve_write(
                keys=pending,
                is_temporary=[False] * len(pending),
                layout_desc=layout_desc,
                tag=_L1_WRITE_TAG,
            )
            reserve_result.update(results)
            # Retry only allocation failures on the next candidate. Terminal
            # conflicts stay attributed to the manager that reported them.
            pending = [
                key for key in pending if results[key][0] == L1Error.OUT_OF_MEMORY
            ]

        successful_keys = [
            key
            for key, (_, memory_obj) in reserve_result.items()
            if memory_obj is not None
        ]
        failed_keys = [
            k for k, (_, memory_obj) in reserve_result.items() if memory_obj is None
        ]
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
            key
            for key, (error, _) in reserve_result.items()
            if error == L1Error.OUT_OF_MEMORY
        ]
        if oom_keys:
            self._event_bus.publish(
                Event(
                    event_type=EventType.L1_ALLOCATION_FAILED,
                    metadata={"during": "l1_store", "keys": oom_keys},
                )
            )

        return reserve_result

    def _build_capacities(self) -> list[ModuleMemoryCapacity]:
        """Assemble one capacity entry per memory compartment.

        Returns:
            L1 per backing medium, then one entry per L2 adapter.
        """
        per_backend: dict[L1BackendType, int] = defaultdict(int)
        for manager in self._l1_managers_by_id.values():
            for backend, size in manager.get_capacity_bytes_by_backend().items():
                per_backend[backend] += size
        capacities = [
            ModuleMemoryCapacity(
                tier=Tier.L1,
                backend=backend.value,
                capacity_bytes=configured,
                shared=False,
            )
            for backend, configured in per_backend.items()
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
        usages = [m.get_memory_usage() for m in self._l1_managers_by_id.values()]
        return sum(u for u, _ in usages), sum(t for _, t in usages)

    # L1 reconfiguration APIs
    def get_l1_devdax_arena_statuses(self) -> list[DevDaxArenaStatus]:
        """Return runtime status for every Device-DAX L1 arena.

        Returns:
            One status per mapped arena, in pool order.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed.
            ValueError: This legacy operation requires a single L1 manager.
        """
        self._require_single_l1()
        return self._l1_manager.get_devdax_arena_statuses()

    def add_l1_devdax_device(
        self,
        device_path: str,
        size_in_bytes: int,
    ) -> DevDaxArenaStatus:
        """Add a Device-DAX device to the L1 arena pool.

        A successful addition publishes the current whole capacity topology.

        The addition is refused while an L2 adapter that registers a single L1
        memory region is configured.
        The compatibility checks and addition are protected by ``_lifecycle_lock``.

        Args:
            device_path: Path of the Device-DAX device to map.
            size_in_bytes: Number of bytes to map.

        Returns:
            Status of the newly added arena.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed, a single-region
                L2 adapter is configured (409), the physical device is already
                mapped by L2 (409), or the request cannot be applied.
            ValueError: This legacy operation requires a single L1 manager.
        """
        self._require_single_l1()
        with self._lifecycle_lock:
            # Report the L1 backing error before adapter compatibility.
            self._l1_manager.get_devdax_arena_statuses()
            incompatible = self._single_region_adapter_names()
            if incompatible:
                raise L1ReconfigureError(
                    409,
                    "cannot add a Device-DAX L1 arena: L2 adapters that "
                    "register a single L1 memory region are configured "
                    f"({', '.join(incompatible)}); their transfers cover "
                    "only the primary arena",
                )
            device_owners = self._l2_device_owner_names(device_path)
            if device_owners:
                raise L1ReconfigureError(
                    409,
                    "cannot add a Device-DAX L1 arena: the physical device "
                    "is already mapped by L2 adapter(s) "
                    f"({', '.join(device_owners)})",
                )
            status = self._l1_manager.add_devdax_device(device_path, size_in_bytes)
        self._publish_capacity_changed()
        return status

    def remove_l1_devdax_device(
        self,
        device_path: str,
        mode: DevDaxRemoveMode = DevDaxRemoveMode.DRAIN,
    ) -> DevDaxArenaStatus:
        """Remove a Device-DAX device from the L1 arena pool.

        Whenever the call changes usable capacity, it publishes the current
        whole topology. This includes a drain transition whose later device
        cleanup raises an exception.

        Args:
            device_path: Path of the mapped Device-DAX device.
            mode: Removal strategy. Only drain mode is currently supported.

        Returns:
            Status of the arena after the removal request.

        Raises:
            L1ReconfigureError: If L1 is not Device-DAX backed or the request
                cannot be applied.
            RuntimeError: If device synchronization or cleanup fails after the
                drain transition.
            OSError: If unmapping or closing the device fails after the drain
                transition.
            ValueError: This legacy operation requires a single L1 manager.
        """
        self._require_single_l1()
        target_was_active = self._l1_devdax_arena_is_active(device_path)
        try:
            status = self._l1_manager.remove_devdax_device(device_path, mode)
        except Exception:
            # Draining begins before an empty arena is synchronized and
            # unmapped. If that cleanup raises, usable capacity has still
            # changed and the coordinator must not retain the old topology.
            try:
                if target_was_active and not self._l1_devdax_arena_is_active(
                    device_path
                ):
                    self._publish_capacity_changed()
            except Exception:
                logger.exception(
                    "Failed to reconcile L1 capacity after a Device-DAX remove error"
                )
            raise
        self._publish_capacity_changed()
        return status

    def _single_region_adapter_names(self) -> list[str]:
        """Return type names of registered L2 adapters needing one L1 region.

        The caller holds ``_lifecycle_lock`` so the answer stays valid while
        it acts on it; ``_adapters_lock`` only guards the dict read.
        """
        with self._adapters_lock:
            descriptors = list(self._adapter_descriptors.values())
        return [
            name
            for descriptor in descriptors
            if (name := requires_single_l1_memory_region(descriptor.config)) is not None
        ]

    def _l1_devdax_arena_is_active(self, device_path: str) -> bool:
        """Return whether an ACTIVE Device-DAX arena is mapped at ``device_path``.

        The path must match the one used when adding the arena. ``False``
        when L1 is not Device-DAX backed or nothing is registered there.
        """
        try:
            status = self._l1_manager.get_devdax_arena_status(device_path)
        except L1ReconfigureError:
            return False
        return status.state is DevDaxArenaState.ACTIVE

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

        Snapshot construction and enqueue are serialized because registration
        can publish concurrently with runtime reconfiguration. The subscriber
        assigns revisions in queue order, so an older snapshot must not be
        enqueued after a newer one. The locks used by ``_build_capacities`` to
        protect its reads are still required; this lock only orders declarations.

        The event carries the whole topology, not a delta, so a later
        publication can repair a dropped declaration.
        """
        with self._capacity_publish_lock:
            snapshot = CapacitySnapshot(modules=tuple(self._build_capacities()))
            self._event_bus.publish(
                Event(
                    event_type=EventType.SM_CAPACITY_CHANGED,
                    metadata={"snapshot": snapshot},
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
        with self._lifecycle_lock if operation == "add" else nullcontext():
            adapter = self._get_reconfigurable_l2_adapter(adapter_index)
            result = adapter.reconfigure(
                operation,
                payload,
                device_owners=lambda path: self._device_owner_names(path, adapter),
            )
        result["adapter_index"] = adapter_index
        self._publish_capacity_changed()
        return result

    def add_l2_adapter(self, config: L2AdapterConfigBase) -> int:
        """Blocking function to add a new L2 adapter at runtime. Thread-safe.

        Args:
            config: The adapter configuration.

        Returns:
            The stable id assigned to the new adapter.

        Raises:
            ValueError: If the adapter registers a single L1 memory region
                while L1 spans more than one (hybrid DRAM + Device-DAX, or
                more than one Device-DAX arena), or a DAX device is already
                mapped by L1 or another L2 adapter.
        """
        with self._lifecycle_lock:
            # Mirror of the check in add_l1_devdax_device: a single-region
            # adapter may only be added while L1 is exactly one memory region.
            adapter_name = requires_single_l1_memory_region(config)
            if config.affinity_tag not in self._l1_by_tag:
                raise ValueError(f"Unknown L1 affinity_tag: {config.affinity_tag}")
            if self._l1_by_tag[config.affinity_tag].config.gds_l1_config is not None:
                raise ValueError("L2 affinity requires a host-backed DRAM or DEVDAX L1")
            region_count = self._l1_by_tag[config.affinity_tag].memory_region_count()
            if adapter_name is not None and region_count > 1:
                raise ValueError(
                    f"{adapter_name} registers a single L1 memory region, but "
                    f"L1 currently spans {region_count} regions (hybrid DRAM + "
                    "Device-DAX, or more than one Device-DAX arena); remove the "
                    "additional Device-DAX regions before adding it"
                )
            # Check all DAX devices before the constructor maps any.
            device_config = unwrap_l2_adapter_config(config)
            if get_type_name_for_config(device_config) == "dax":
                for device in cast(Any, device_config).devices:
                    owners = self._device_owner_names(device.device_path)
                    if owners:
                        raise ValueError(
                            f"device {device.device_path} is already mapped by "
                            f"{', '.join(owners)}"
                        )
            adapter_id, adapter, descriptor = self._build_l2_adapter(config)
            for listener in self._registered_l2_listeners:
                adapter.register_listener(listener)
            with self._adapters_lock:
                self._l2_adapters[adapter_id] = adapter
                self._adapter_descriptors[adapter_id] = descriptor
            owner = self._l1_by_tag[config.affinity_tag].l1_manager_id
            self._store_controllers[owner].add_adapter(adapter_id, adapter, descriptor)
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
            owner = self._l1_by_tag[affinity].l1_manager_id
            store_done = self._store_controllers[owner].request_remove_adapter(
                adapter_id
            )
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
        for manager in self._l1_managers_by_id.values():
            manager.clear(force=force)

    def close(self):
        """
        Close the storage manager and release all resources.
        """
        self._prefetch_controller.stop()
        for controller in self._store_controllers.values():
            controller.stop()
        for controller in self._eviction_controllers:
            controller.stop()
        self._l2_eviction_controller.stop()

        PeriodicEventNotifier.shutdown()

        for adapter in self._l2_adapters.values():
            adapter.close()

        for manager in self._l1_managers_by_id.values():
            manager.close()

    @property
    def is_multi_l1(self) -> bool:
        """Whether owner metadata is required to resolve a serving read."""
        return len(self._l1_managers_by_id) > 1

    def report_status(self) -> dict:
        """Return per-L1 status and aggregate health; retain single-L1 field names."""
        managers = list(self._l1_managers_by_id.values())
        tags = [c.tag for c in self._l1_configs]
        l1s = {tag: m.report_status() for tag, m in zip(tags, managers, strict=True)}
        capacities: dict[str, int] = defaultdict(int)
        for status in l1s.values():
            for backend, size in status["capacity_bytes_by_backend"].items():
                capacities[backend] += size
        stores = {
            tag: self._store_controllers[m.l1_manager_id].report_status()
            for tag, m in zip(tags, managers, strict=True)
        }
        evictions = {
            tag: c.report_status()
            for tag, c in zip(tags, self._eviction_controllers, strict=True)
        }
        prefetch = self._prefetch_controller.report_status()
        l2_eviction = self._l2_eviction_controller.report_status()
        adapters = [a.report_status() for _id, _desc, a in self._snapshot_adapters()]
        result = {
            "is_healthy": all(
                c["is_healthy"]
                for c in [
                    *l1s.values(),
                    *stores.values(),
                    *evictions.values(),
                    prefetch,
                    l2_eviction,
                    *adapters,
                ]
            ),
            "l1_managers": l1s,
            "store_controllers": stores,
            "l1_eviction_controllers": evictions,
            "prefetch_controller": prefetch,
            "l2_eviction_controller": l2_eviction,
            "l2_adapters": adapters,
            "num_l2_adapters": len(adapters),
            "l1_usage": self.get_l1_usage(),
            "l1_capacity_bytes_by_backend": dict(capacities),
        }
        if not self.is_multi_l1:
            result.update(
                l1_manager=l1s[tags[0]],
                store_controller=stores[tags[0]],
                l1_eviction_controller=evictions[tags[0]],
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
        Check memory consistency in every L1 manager.

        Returns:
            True if memory is consistent, False otherwise.
        """
        checks = [m.memcheck() for m in self._l1_managers_by_id.values()]
        return all(checks)

    def prepare_read_completion(
        self,
        keys: list[ObjectKey],
        l1_owners: dict[ObjectKey, int] | None = None,
    ) -> L1WriteCompletion:
        """Capture exact read owners for stream-ordered completion.

        Args:
            keys: Keys whose read locks the caller owns from prefetch.
            l1_owners: Owners retained by that prefetch; required for multiple L1s.

        Returns:
            Serializable owner/key groups, identical in shape to write completion.

        Raises:
            ValueError: A key has no registered owner, or multi-L1 owners are absent.
        """
        return list(self._read_groups(keys, l1_owners).items())

    def finish_read_by_owner(self, completion: L1WriteCompletion) -> None:
        """Release one read lock per key on its captured owner after device use.

        Args:
            completion: Groups produced by prepare_read_completion.

        Raises:
            ValueError: An owner is not registered.
        """
        if any(owner not in self._l1_managers_by_id for owner, _ in completion):
            raise ValueError("Read completion requires a registered L1 owner")
        if not self.is_multi_l1:
            # Keep single-L1 traces portable across process-local owner IDs.
            self.finish_read_prefetched([key for _, keys in completion for key in keys])
            return
        for owner, keys in completion:
            self.finish_read_prefetched(keys, l1_owners=dict.fromkeys(keys, owner))

    def _read_groups(
        self, keys: list[ObjectKey], owners: dict[ObjectKey, int] | None
    ) -> dict[int, list[ObjectKey]]:
        if owners is None:
            self._require_single_l1()
            return {self._l1_manager.l1_manager_id: keys}
        groups: dict[int, list[ObjectKey]] = defaultdict(list)
        for key in keys:
            if key not in owners or owners[key] not in self._l1_managers_by_id:
                raise ValueError("Read requires the exact owner retained by prefetch")
            groups[owners[key]].append(key)
        return groups

    def _read_objects(
        self, keys: list[ObjectKey], owners: dict[ObjectKey, int] | None
    ) -> dict[ObjectKey, L1OperationResult]:
        results: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.KEY_NOT_EXIST, None) for key in keys
        }
        for owner, group in self._read_groups(keys, owners).items():
            results.update(self._l1_managers_by_id[owner].unsafe_read(group))
        return results

    def _finish_read_objects(
        self, keys: list[ObjectKey], count: int, owners: dict[ObjectKey, int] | None
    ) -> dict[ObjectKey, L1Error]:
        results = {}
        for owner, group in self._read_groups(keys, owners).items():
            results.update(
                self._l1_managers_by_id[owner].finish_read(group, read_locks=count)
            )
        return results

    def _require_single_l1(self) -> None:
        """Guard legacy operations that cannot identify an individual L1."""
        if len(self._l1_managers_by_id) != 1:
            raise ValueError(
                "This operation requires a single L1; use owner-routed operations"
            )

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
        if config.affinity_tag not in self._l1_by_tag:
            raise ValueError(f"Unknown L1 affinity_tag: {config.affinity_tag}")
        manager = self._l1_by_tag[config.affinity_tag]
        adapter_id = self._next_adapter_id
        self._next_adapter_id += 1
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

    def _device_owner_names(
        self, device_path: str, exclude: Optional[L2ReconfigurableAdapter] = None
    ) -> list[str]:
        """Return other L1/L2 owners while the caller holds the lifecycle lock."""
        owners = [
            f"L1[{tag}]" if self.is_multi_l1 else "L1"
            for tag, manager in self._l1_by_tag.items()
            if manager.owns_device(device_path)
        ]
        return owners + self._l2_device_owner_names(device_path, exclude)

    def _l2_device_owner_names(
        self, device_path: str, exclude: Optional[L2ReconfigurableAdapter] = None
    ) -> list[str]:
        """Return L2 type names that own the physical device at a path.

        The caller holds ``_lifecycle_lock`` so registered adapters cannot be
        added or deleted between this check and the mapping attempt.

        Args:
            device_path: Candidate Device-DAX path.
            exclude: L2 adapter whose own mappings are ignored.

        Returns:
            Registered adapter type names whose open device has the same
            physical identity.
        """
        owners: list[str] = []
        for _adapter_id, descriptor, adapter in self._snapshot_adapters():
            owner = self._unwrap_reconfigurable_l2_adapter(adapter)
            if (
                owner is not exclude
                and isinstance(owner, L2DeviceOwner)
                and owner.owns_device(device_path)
            ):
                owners.append(descriptor.type_name)
        return owners

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
