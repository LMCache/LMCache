# SPDX-License-Identifier: Apache-2.0
"""Remote binding of the L1 lifecycle: a region a memory orchestrator owns."""

# Standard
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from mmap import PAGESIZE
import threading
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import get_size_bytes
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    DRAML1ManagerConfig,
    L1ManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import (
    L1ManagerInterface,
    L1ManagerListener,
    L1MemoryDesc,
    L1ObjectMeta,
    L1OperationResult,
)
from lmcache.v1.distributed.l1_manager import next_l1_manager_id, validate_read_locks
from lmcache.v1.distributed.l1_metrics import register_l1_metrics, unregister_l1_metrics
from lmcache.v1.distributed.shared_l1.media import RegionMedium
from lmcache.v1.distributed.shared_l1.visibility import Visibility
from lmcache.v1.distributed.shared_l1.wire import (
    layout_from_wire,
    layout_to_wire,
)
from lmcache.v1.memory_management import TensorMemoryObj
from lmcache.v1.memory_orchestrator.api import (
    DEFAULT_LAYOUT_FINGERPRINT,
    OrchestratorError,
    OrchestratorUnavailableError,
    ReadRequest,
    ReadStatus,
    RegionFencedError,
    RegionUsage,
    RequestRejectedError,
    TokenStatus,
    WriteRequest,
    WriteStatus,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus

logger = init_logger(__name__)

# One Usage RPC per this many seconds serves the eviction and status pollers.
_USAGE_TTL_SECONDS = 1.0

# After the orchestrator was unreachable, lookups, reservations and usage polls
# skip the RPC for this long instead of each paying the client's retries.
_UNREACHABLE_COOLDOWN_SECONDS = 1.0


@dataclass
class _WriteGrant:
    """A write extent this server reserved and has not committed or aborted."""

    token: bytes
    memory_obj: TensorMemoryObj


@dataclass
class _ReadHold:
    """Read leases this server holds on one object, and its view."""

    memory_obj: TensorMemoryObj
    leases: list[bytes] = field(default_factory=list)


class SharedL1Manager(L1ManagerInterface):
    """An L1 whose objects live in a region a memory orchestrator owns.

    It holds only its own write grants (until commit or abort) and read leases.

    Args:
        config: An L1 configuration with a ``shared`` section and a
            unique ``client_id``; its type picks the medium.

    Raises:
        ValueError: The configuration is incomplete or does not match the
            orchestrator's region.
        RuntimeError: The orchestrator wants an offline reset, or visibility
            is unsupported on this host.
        OrchestratorError: The orchestrator refused or could not be reached.
        OSError: The medium cannot be mapped.
    """

    def __init__(self, config: L1ManagerConfig) -> None:
        shared = config.shared
        if shared is None:
            raise ValueError("A shared L1 needs a shared section with a client_id")
        if isinstance(config, DevDaxL1ManagerConfig):
            backend = L1BackendType.DEVDAX
        elif isinstance(config, DRAML1ManagerConfig):
            backend = L1BackendType.DRAM
        else:
            raise ValueError("A shared L1 requires a DRAM or DEVDAX configuration")
        self._config = config
        self._shared = shared
        self._l1_manager_id = next_l1_manager_id()
        self._lock = threading.Lock()
        self._listeners: list[L1ManagerListener] = []
        self._event_bus = get_event_bus()
        self._client = MemoryOrchestratorClient(
            shared.orchestrator,
            shared.region_id,
            shared.client_id,
            rpc_timeout_s=shared.rpc_timeout_seconds,
        )
        with ExitStack() as cleanup:
            cleanup.callback(self._client.close)
            contract = self._client.describe_region()
            if contract.reset_required:
                raise RuntimeError(
                    f"Memory orchestrator for region {shared.region_id!r} needs "
                    "an offline reset: stop every server using the region, "
                    "stop the orchestrator, delete its startup marker, restart"
                )
            if contract.capacity_bytes > config.memory_config.size_in_bytes:
                raise ValueError(
                    f"Region {shared.region_id!r} holds {contract.capacity_bytes} "
                    "bytes, more than this L1's size_gb allows "
                    f"({config.memory_config.size_in_bytes} bytes)"
                )
            if (
                contract.visibility_mode == "software_fenced"
                and contract.alignment_bytes < PAGESIZE
            ):
                # Different writers must never write back the same cache line.
                raise ValueError("software_fenced extents require page alignment")
            visibility = Visibility(contract.visibility_mode)
            cleanup.callback(visibility.close)
            medium = RegionMedium(
                shared.path, contract.capacity_bytes, contract.alignment_bytes, backend
            )
            cleanup.callback(medium.close)
            self._client.register(
                layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
                mapped_bytes=medium.capacity_bytes,
                visibility_mode=visibility.mode,
            )
            if self._client.region_epoch != contract.region_epoch:
                raise RegionFencedError(
                    "The region changed while its medium was mapped"
                )
            cleanup.pop_all()
        self._medium = medium
        self._visibility = visibility
        self._writes: dict[tuple[ObjectKey, str], _WriteGrant] = {}
        self._reads: dict[ObjectKey, _ReadHold] = {}
        self._staging_bytes = 0
        self._reachable = True
        self._retry_after = 0.0
        self._fenced_reason: str | None = None
        self._usage: RegionUsage | None = None
        self._usage_time = 0.0
        register_l1_metrics(self)
        logger.info(
            "Shared L1 %r mapped region %r (%d bytes, epoch %d, %s) through %s "
            "%s (%s) as client %r",
            config.tag,
            shared.region_id,
            contract.capacity_bytes,
            self._client.region_epoch,
            visibility.mode,
            medium.backend_type.value,
            medium.path,
            medium.data_path,
            shared.client_id,
        )

    @property
    def l1_manager_id(self) -> int:
        """Return this manager's process-local owner tag."""
        return self._l1_manager_id

    @property
    def config(self) -> L1ManagerConfig:
        """Return this manager's configuration."""
        return self._config

    def register_listener(self, listener: L1ManagerListener) -> None:
        """Register a listener for this L1's lifecycle notifications."""
        with self._lock:
            self._listeners.append(listener)

    def reserve_write(
        self,
        keys: list[ObjectKey],
        is_temporary: list[bool],
        layout_desc: MemoryLayoutDesc,
        tag: str = "",
    ) -> dict[ObjectKey, L1OperationResult]:
        """Ask the orchestrator for one extent per key.

        Args:
            keys: Keys to write.
            is_temporary: One flag per key; shared objects are never temporary.
            layout_desc: Layout of every object in the batch.
            tag: The writer's identity; the finish calls must pass the same.

        Returns:
            Per key ``(SUCCESS, view)``, ``(KEY_NOT_WRITABLE, None)`` when any
            server (this tag included) committed or is writing the key, or
            ``(OUT_OF_MEMORY, None)`` when the region is full or the orchestrator
            cannot be used, so ordered overflow tries the next L1.

        Raises:
            ValueError: The lengths differ, or a temporary object is requested.
            RequestRejectedError: A key's layout conflicts with the shared object.
        """
        if len(keys) != len(is_temporary):
            raise ValueError(
                f"reserve_write: {len(keys)} keys but "
                f"{len(is_temporary)} is_temporary flags"
            )
        if any(is_temporary):
            raise ValueError("A shared L1 holds no temporary objects")
        ret: dict[ObjectKey, L1OperationResult] = {}
        with self._lock:
            for key in keys:
                if (key, tag) in self._writes:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
        pending = [key for key in keys if key not in ret]
        if not pending or self._skip_request_rpc():
            ret.update((key, (L1Error.OUT_OF_MEMORY, None)) for key in pending)
            return ret
        payload_bytes = get_size_bytes(layout_desc.shapes, layout_desc.dtypes)
        wire = layout_to_wire(layout_desc)
        try:
            grants = self._client.reserve_write(
                [WriteRequest(key, payload_bytes, wire) for key in pending]
            )
        except RequestRejectedError:
            # Invalid metadata must not silently overflow into a private L1.
            raise
        except OrchestratorError as exc:
            self._on_rpc_error("ReserveWrite", exc)
            ret.update((key, (L1Error.OUT_OF_MEMORY, None)) for key in pending)
            return ret
        self._on_rpc_ok()
        granted: list[ObjectKey] = []
        bad_tokens: list[bytes] = []
        with self._lock:
            for key, grant in zip(pending, grants, strict=True):
                if grant.status is WriteStatus.OUT_OF_SPACE:
                    ret[key] = (L1Error.OUT_OF_MEMORY, None)
                    continue
                if grant.status is not WriteStatus.WRITE_GRANTED:
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                if grant.handle is None or grant.token is None:
                    logger.error("Shared L1: write grant has no extent or token")
                    if grant.token is not None:
                        bad_tokens.append(grant.token)
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                try:
                    memory_obj = self._medium.view(
                        grant.handle.offset, grant.handle.length, layout_desc
                    )
                except ValueError:
                    logger.exception("Shared L1: orchestrator granted a bad extent")
                    bad_tokens.append(grant.token)
                    ret[key] = (L1Error.KEY_NOT_WRITABLE, None)
                    continue
                memory_obj.set_l1_manager(self._l1_manager_id)
                self._writes[(key, tag)] = _WriteGrant(grant.token, memory_obj)
                self._staging_bytes += payload_bytes
                ret[key] = (L1Error.SUCCESS, memory_obj)
                granted.append(key)
        if bad_tokens:
            self._abort_tokens(bad_tokens)
        for listener in self._snapshot_listeners():
            listener.on_l1_keys_reserved_write(granted)
        self._publish(EventType.L1_WRITE_RESERVED, {"keys": granted, "tag": tag})
        return ret

    def finish_write(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Publish ``tag``'s written extents and commit them.

        The caller must have waited for the device-to-host copies.

        Args:
            keys: Keys whose extents were written.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key ``SUCCESS`` once committed, ``KEY_NOT_EXIST`` when ``tag``
            reserved nothing for the key, ``KEY_IN_WRONG_STATE`` when the
            orchestrator refused the commit or could not be asked (the store
            fails; a later writer may find the key committed after all).
        """
        ret, done, meta = self._commit(keys, tag)
        if done:
            for listener in self._snapshot_listeners():
                listener.on_l1_keys_write_finished(done)
            self._publish(EventType.L1_WRITE_FINISHED, {"keys": done, "meta": meta})
        return ret

    def finish_write_and_reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1, tag: str = ""
    ) -> dict[ObjectKey, L1OperationResult]:
        """Commit ``tag``'s extents, then take read leases on them.

        Two round trips; safe since committed objects stay until a region reset.

        Args:
            keys: Keys whose extents were written.
            read_locks: Read leases to take per committed key.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key ``(SUCCESS, view)`` with leases taken, else the commit or
            read error and None.
        """
        errors, done, meta = self._commit(keys, tag)
        ret: dict[ObjectKey, L1OperationResult] = {
            key: (error, None)
            for key, error in errors.items()
            if error is not L1Error.SUCCESS
        }
        if done:
            ret.update(self._reserve_read(done, read_locks, notify=False))
            for listener in self._snapshot_listeners():
                listener.on_l1_keys_finish_write_and_reserve_read(done)
            self._publish(
                EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
                {"keys": done, "meta": meta},
            )
        return ret

    def finish_write_and_delete(
        self, keys: list[ObjectKey], tag: str = ""
    ) -> dict[ObjectKey, L1Error]:
        """Abort ``tag``'s write reservations; the keys become writable again.

        Args:
            keys: Keys whose reservations to abort.
            tag: The writer's tag passed to ``reserve_write``.

        Returns:
            Per key ``SUCCESS``, ``KEY_NOT_EXIST`` when ``tag`` reserved
            nothing for the key, or ``KEY_IN_WRONG_STATE`` when the
            orchestrator refused or could not be asked.
        """
        ret, taken = self._take_writes(keys, tag)
        if not taken:
            return ret
        statuses = self._abort_tokens([grant.token for _, grant in taken])
        for (key, _), status in zip(taken, statuses, strict=True):
            ok = status is TokenStatus.OK
            ret[key] = L1Error.SUCCESS if ok else L1Error.KEY_IN_WRONG_STATE
        gone = [key for key, _ in taken]
        for listener in self._snapshot_listeners():
            listener.on_l1_keys_deleted_by_manager(gone)
        return ret

    def reserve_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1OperationResult]:
        """Take ``read_locks`` read leases per committed key.

        Args:
            keys: Keys to read.
            read_locks: Leases per key, one per reader of the object.

        Returns:
            Per key ``(SUCCESS, view)`` with the leases held, else
            ``(KEY_NOT_EXIST, None)``: missing, still being written, or the
            orchestrator cannot be used.
        """
        return self._reserve_read(keys, read_locks, notify=True)

    def unsafe_read(self, keys: list[ObjectKey]) -> dict[ObjectKey, L1OperationResult]:
        """Return leased views; unleased keys return ``KEY_NOT_EXIST``."""
        with self._lock:
            ret: dict[ObjectKey, L1OperationResult] = {}
            for key in keys:
                hold = self._reads.get(key)
                if hold is None:
                    ret[key] = (L1Error.KEY_NOT_EXIST, None)
                else:
                    ret[key] = (L1Error.SUCCESS, hold.memory_obj)
            return ret

    def finish_read(
        self, keys: list[ObjectKey], read_locks: int = 1
    ) -> dict[ObjectKey, L1Error]:
        """Release ``read_locks`` leases per key.

        Args:
            keys: Keys to release.
            read_locks: Leases to release per key.

        Returns:
            Per key ``SUCCESS``, ``KEY_NOT_EXIST`` when no lease is held, or
            ``KEY_IN_WRONG_STATE`` when the orchestrator refused a lease or
            could not be asked (the lease is dropped locally either way).
        """
        count = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1Error] = {}
        released: list[tuple[ObjectKey, list[bytes]]] = []
        with self._lock:
            for key in keys:
                hold = self._reads.get(key)
                if hold is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                released.append((key, hold.leases[-count:]))
                del hold.leases[-count:]
                if not hold.leases:
                    del self._reads[key]
        if not released:
            return ret
        tokens = [token for _, key_tokens in released for token in key_tokens]
        try:
            statuses = self._client.finish_read(tokens)
            self._on_rpc_ok()
        except OrchestratorError as exc:
            self._on_rpc_error("FinishRead", exc)
            statuses = [TokenStatus.STALE_TOKEN] * len(tokens)
        position = 0
        done: list[ObjectKey] = []
        for key, key_tokens in released:
            key_statuses = statuses[position : position + len(key_tokens)]
            position += len(key_tokens)
            if all(status is TokenStatus.OK for status in key_statuses):
                ret[key] = L1Error.SUCCESS
                done.append(key)
            else:
                ret[key] = L1Error.KEY_IN_WRONG_STATE
        for listener in self._snapshot_listeners():
            listener.on_l1_keys_read_finished(done)
        self._publish(EventType.L1_READ_FINISHED, {"keys": done})
        return ret

    def delete(
        self, keys: list[ObjectKey], force: bool = False
    ) -> dict[ObjectKey, L1Error]:
        """Return ``KEY_IS_LOCKED`` for every key, including forced deletes.

        Shared objects may still be read by other servers.
        """
        return {key: L1Error.KEY_IS_LOCKED for key in keys}

    def clear(self, force: bool = False) -> None:
        """Leave shared objects in place, including when ``force`` is set."""
        logger.info(
            "Shared L1 %r: clear leaves shared objects in place", self._config.tag
        )

    def touch_keys(self, keys: list[ObjectKey]) -> None:
        """Notify listeners of accessed keys without changing remote state."""
        for listener in self._snapshot_listeners():
            listener.on_l1_keys_accessed(keys)

    def is_key_evictable(self, key: ObjectKey) -> bool:
        """Return False: shared objects are never evicted here."""
        return False

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(allocated_bytes, capacity_bytes)`` of the whole region.

        Allocated bytes include extents of aborted writes and are at most one
        second old: the last known value, or zero, when the orchestrator cannot
        be asked.
        """
        usage = self._region_usage()
        used = usage.allocated_bytes if usage is not None else 0
        return used, self._medium.capacity_bytes

    def get_staging_memory_usage(self) -> int:
        """Return payload bytes this server reserved and has not committed."""
        with self._lock:
            return self._staging_bytes

    def get_capacity_bytes_by_backend(self) -> dict[L1BackendType, int]:
        """Return the region capacity under the medium's backend type."""
        return {self._medium.backend_type: self._medium.capacity_bytes}

    def get_l1_memory_desc(self) -> L1MemoryDesc | None:
        """Return None: L2 adapters never register the shared region."""
        return None

    def owns_device(self, device_path: str) -> bool:
        """Return whether the candidate path names this mapped medium."""
        return self._medium.owns_device(device_path)

    def memory_region_count(self) -> int:
        """Return 1: one mapped region."""
        return 1

    def report_status(self) -> dict[str, object]:
        """Return the L1 status fields plus a ``shared`` section.

        ``is_healthy`` is False while the orchestrator is unreachable and for
        good once the manager is fenced.
        """
        usage = self._region_usage()
        used, total = self.get_memory_usage()
        with self._lock:
            write_count = len(self._writes)
            read_count = len(self._reads)
            staging_bytes = self._staging_bytes
            reachable = self._reachable
            fenced = self._fenced_reason
        return {
            "is_healthy": reachable and fenced is None,
            "total_object_count": usage.valid + usage.writing if usage else 0,
            "write_locked_count": write_count,
            "read_locked_count": read_count,
            "temporary_count": 0,
            "staging_object_count": write_count,
            "staging_bytes": staging_bytes,
            "memory_used_bytes": used,
            "memory_total_bytes": total,
            "memory_configured_bytes": total,
            "capacity_bytes_by_backend": {self._medium.backend_type.value: total},
            "memory_usage_ratio": used / total if total > 0 else 0.0,
            # Write expiry belongs to the orchestrator; read leases do not expire.
            "write_ttl_seconds": None,
            "read_ttl_seconds": None,
            "shared": {
                "orchestrator": self._shared.orchestrator,
                "region_id": self._shared.region_id,
                "client_id": self._shared.client_id,
                "region_epoch": self._client.region_epoch,
                "medium": self._medium.backend_type.value,
                "path": self._medium.path,
                "visibility_mode": self._visibility.mode,
                "data_path": self._medium.data_path,
                "orchestrator_reachable": reachable,
                "fenced": fenced,
                "region_usage": None
                if usage is None
                else {k: v for k, v in asdict(usage).items() if k != "capacity_bytes"},
            },
        }

    def memcheck(self) -> bool:
        """Whether the last orchestrator call succeeded and the manager is unfenced."""
        with self._lock:
            return self._reachable and self._fenced_reason is None

    def close(self) -> None:
        """Abort unfinished writes, release leases, leave the region.

        Call only after users of this mapping have drained their copies.
        ``CloseClient`` also retires this client's remaining state. If it
        cannot be reached, read leases remain until this client re-registers;
        unfinished writes also expire under the orchestrator's write TTL.
        """
        unregister_l1_metrics(self)
        with self._lock:
            while self._writes:
                self._writes.popitem()[1].memory_obj.invalidate()
            self._reads.clear()
            self._staging_bytes = 0
        # CloseClient retires all of this incarnation's grants and leases.
        self._client.close()
        self._visibility.close()
        self._medium.close()

    def _take_writes(
        self, keys: list[ObjectKey], tag: str
    ) -> tuple[dict[ObjectKey, L1Error], list[tuple[ObjectKey, _WriteGrant]]]:
        """Pop ``tag``'s grants for ``keys``; ``KEY_NOT_EXIST`` for keys without."""
        ret: dict[ObjectKey, L1Error] = {}
        taken: list[tuple[ObjectKey, _WriteGrant]] = []
        with self._lock:
            for key in keys:
                grant = self._writes.pop((key, tag), None)
                if grant is None:
                    ret[key] = L1Error.KEY_NOT_EXIST
                    continue
                self._staging_bytes -= grant.memory_obj.get_size()
                taken.append((key, grant))
        return ret, taken

    def _commit(
        self, keys: list[ObjectKey], tag: str
    ) -> tuple[dict[ObjectKey, L1Error], list[ObjectKey], list[L1ObjectMeta]]:
        """Publish and commit ``tag``'s extents; return errors, committed keys, meta."""
        ret, taken = self._take_writes(keys, tag)
        statuses = [TokenStatus.STALE_TOKEN] * len(taken)
        if taken and not self._fenced():
            # Push the written bytes out of this host's caches before others read.
            try:
                self._visibility.publish(
                    [
                        (grant.memory_obj.data_ptr, grant.memory_obj.get_size())
                        for _, grant in taken
                    ]
                )
            except Exception:
                self._abort_tokens([grant.token for _, grant in taken])
                raise
            try:
                statuses = self._client.finish_write([g.token for _, g in taken])
                self._on_rpc_ok()
            except OrchestratorError as exc:
                self._on_rpc_error("FinishWrite", exc)
        done: list[ObjectKey] = []
        meta: list[L1ObjectMeta] = []
        backend = self._medium.backend_type
        for (key, grant), status in zip(taken, statuses, strict=True):
            if status is TokenStatus.OK:
                ret[key] = L1Error.SUCCESS
                done.append(key)
                meta.append(L1ObjectMeta(grant.memory_obj.get_size(), backend))
            else:
                ret[key] = L1Error.KEY_IN_WRONG_STATE
        return ret, done, meta

    def _reserve_read(
        self, keys: list[ObjectKey], read_locks: int, notify: bool
    ) -> dict[ObjectKey, L1OperationResult]:
        """Lease committed objects, invalidate their ranges, return views."""
        count = validate_read_locks(read_locks)
        ret: dict[ObjectKey, L1OperationResult] = {
            key: (L1Error.KEY_NOT_EXIST, None) for key in keys
        }
        if not keys or self._skip_request_rpc():
            return ret
        try:
            grants = self._client.reserve_read(
                [ReadRequest(key, count) for key in keys]
            )
        except OrchestratorError as exc:
            self._on_rpc_error("ReserveRead", exc)
            return ret
        self._on_rpc_ok()
        if all(grant.status is not ReadStatus.READ_GRANTED for grant in grants):
            return ret
        bad_leases: list[bytes] = []
        leased: list[ObjectKey] = []
        ranges: list[tuple[int, int]] = []
        with self._lock:
            for key, grant in zip(keys, grants, strict=True):
                if grant.status is not ReadStatus.READ_GRANTED:
                    continue
                if (
                    grant.handle is None
                    or grant.layout is None
                    or len(grant.leases) != count
                ):
                    logger.error(
                        "Shared L1: read grant has no extent, layout or leases"
                    )
                    bad_leases.extend(grant.leases)
                    continue
                try:
                    layout = layout_from_wire(grant.layout)
                    memory_obj = self._medium.view(
                        grant.handle.offset, grant.handle.length, layout
                    )
                    if memory_obj.get_size() != grant.payload_bytes:
                        raise ValueError("Read grant payload does not match its layout")
                    hold = self._reads.get(key)
                    if hold is not None and (
                        hold.memory_obj.data_ptr != memory_obj.data_ptr
                        or hold.memory_obj.get_shapes() != layout.shapes
                        or hold.memory_obj.get_dtypes() != layout.dtypes
                    ):
                        raise ValueError(
                            "A leased shared object changed its extent/layout"
                        )
                except ValueError:
                    logger.exception("Shared L1: orchestrator granted a bad read")
                    bad_leases.extend(grant.leases)
                    continue
                if hold is None:
                    memory_obj.set_l1_manager(self._l1_manager_id)
                    hold = _ReadHold(memory_obj)
                    self._reads[key] = hold
                hold.leases.extend(grant.leases)
                ret[key] = (L1Error.SUCCESS, hold.memory_obj)
                leased.append(key)
                ranges.append((memory_obj.data_ptr, grant.payload_bytes))
        # Drop stale cached lines; the host-to-device copy must read the device.
        if bad_leases:
            try:
                self._client.finish_read(bad_leases)
            except OrchestratorError as exc:
                self._on_rpc_error("FinishRead", exc)
        try:
            self._visibility.acquire(ranges)
        except Exception:
            self.finish_read(leased, count)
            raise
        if notify:
            for listener in self._snapshot_listeners():
                listener.on_l1_keys_reserved_read(leased)
            self._publish(EventType.L1_READ_RESERVED, {"keys": leased})
        return ret

    def _abort_tokens(self, tokens: list[bytes]) -> list[TokenStatus]:
        """AbortWrite; on failure report every token stale."""
        if self._fenced():
            return [TokenStatus.STALE_TOKEN] * len(tokens)
        try:
            statuses = self._client.abort_write(tokens)
        except OrchestratorError as exc:
            self._on_rpc_error("AbortWrite", exc)
            return [TokenStatus.STALE_TOKEN] * len(tokens)
        self._on_rpc_ok()
        return statuses

    def _region_usage(self) -> RegionUsage | None:
        """Return region usage at most ``_USAGE_TTL_SECONDS`` old."""
        now = time.monotonic()
        with self._lock:
            if (
                self._fenced_reason is not None
                or now < self._retry_after
                or now - self._usage_time < _USAGE_TTL_SECONDS
            ):
                return self._usage
            self._usage_time = now
        try:
            usage = self._client.usage()
        except OrchestratorError as exc:
            self._on_rpc_error("Usage", exc)
        else:
            self._on_rpc_ok()
            with self._lock:
                self._usage = usage
        with self._lock:
            return self._usage

    def _on_rpc_ok(self) -> None:
        with self._lock:
            was_reachable = self._reachable
            self._reachable = True
        if not was_reachable:
            logger.info("Shared L1 %r: orchestrator reachable again", self._config.tag)

    def _on_rpc_error(self, rpc: str, exc: OrchestratorError) -> None:
        """Record a failed RPC; a fencing error disables the manager for good."""
        with self._lock:
            if isinstance(exc, RegionFencedError):
                if self._fenced_reason is not None:
                    return
                self._fenced_reason = f"{rpc}: {exc}"
                logger.error(
                    "Shared L1 %r is fenced (%s). It serves nothing until this "
                    "server restarts after the region is reset.",
                    self._config.tag,
                    self._fenced_reason,
                )
                return
            if isinstance(exc, OrchestratorUnavailableError):
                self._retry_after = time.monotonic() + _UNREACHABLE_COOLDOWN_SECONDS
            was_reachable = self._reachable
            self._reachable = False
        if was_reachable:
            logger.warning(
                "Shared L1 %r: %s failed (%s); reads miss and writes overflow",
                self._config.tag,
                rpc,
                exc,
            )

    def _fenced(self) -> bool:
        with self._lock:
            return self._fenced_reason is not None

    def _skip_request_rpc(self) -> bool:
        """Whether a lookup or reservation skips the RPC: fenced, or cooling down.

        Commits, aborts and releases ignore the cooldown as they change remote state.
        """
        with self._lock:
            return (
                self._fenced_reason is not None or time.monotonic() < self._retry_after
            )

    def _snapshot_listeners(self) -> list[L1ManagerListener]:
        with self._lock:
            return list(self._listeners)

    def _publish(self, event_type: EventType, metadata: dict[str, object]) -> None:
        # ``shared`` keeps region-owned placements out of per-server cache events.
        self._event_bus.publish(
            Event(
                event_type=event_type,
                metadata={"l1_tag": self._config.tag, "shared": True, **metadata},
            )
        )
