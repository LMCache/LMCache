# SPDX-License-Identifier: Apache-2.0
"""P2PController: peer discovery, adapter lifecycle, and lookup serving."""

# Standard
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Literal
import threading
import time

# Third Party
import httpx

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchHandle,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.l2_adapters.p2p_l2_adapter import P2PL2AdapterConfig
from lmcache.v1.distributed.transfer_channel import (
    delete_transfer_channel_context,
    initialize_transfer_channel_context,
)
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.multiprocess.config import CoordinatorConfig, P2PConfig
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType
from lmcache.v1.multiprocess.request_handler import request_handler
from lmcache.v1.periodic_thread import (
    PeriodicThread,
    ThreadLevel,
    ThreadRunSummary,
    create_periodic_thread,
)

logger = init_logger(__name__)

# Sentinel address for keys that were not found (or fell past the L1 prefix).
_INVALID_ADDRESS = TransferChannelAddress(offset=-1, size=0)

# Consecutive missed polls a peer may be absent before its adapter is removed.
_MAX_MISSES = 3


def _group_keys_by_object_group(
    keys: list[ObjectKey],
    group_layout_descs: dict[int, MemoryLayoutDesc],
) -> list[GroupedObjectKeys]:
    """Bucket a flat key list into one key row per object group.

    Each object group becomes a single row holding its keys in request
    order (kv ranks mixed), paired with that group's layout.

    Args:
        keys: The keys received over RPC, in request order.
        group_layout_descs: Memory layout per object group id.

    Returns:
        One :class:`GroupedObjectKeys` per object group, in first-seen order;
        empty when ``keys`` is empty or the object groups received different
        numbers of keys (logged as an error).

    Raises:
        ValueError: If a key's object group has no entry in
            ``group_layout_descs``.

    Note:
        The groups carry no chunk layout, so they are only valid for
        ``"full"`` fetching.
    """
    by_group: dict[int, list[ObjectKey]] = {}
    for key in keys:
        by_group.setdefault(key.object_group_id, []).append(key)
    group_sizes = {gid: len(group_keys) for gid, group_keys in by_group.items()}
    if len(set(group_sizes.values())) > 1:
        logger.error(
            "P2P lookup: object groups received different numbers of keys "
            "(%s); treating all %d keys as misses",
            group_sizes,
            len(keys),
        )
        return []
    rows: list[GroupedObjectKeys] = []
    for object_group_id, group_keys in by_group.items():
        layout_desc = group_layout_descs.get(object_group_id)
        if layout_desc is None:
            raise ValueError(
                f"P2P lookup: no layout for object group {object_group_id} "
                f"(have {sorted(group_layout_descs)})"
            )
        rows.append(
            GroupedObjectKeys(
                keys=group_keys,
                object_group_id=object_group_id,
                layout_desc=layout_desc,
            )
        )
    return rows


class _P2PState(Enum):
    """Registration state of this instance as seen by the coordinator."""

    REGISTERED = "registered"
    DISCONNECTED = "disconnected"
    UNREGISTERED = "unregistered"


@dataclass
class _PeerInstance:
    """A peer's P2P-relevant connection info parsed from ``GET /instances``."""

    instance_id: str
    ip: str
    p2p_advertised_url: str
    mq_port: int


@dataclass
class _PeerAdapter:
    """Bookkeeping for a P2P L2 adapter created for a single peer."""

    adapter_id: int
    ip: str
    p2p_advertised_url: str
    mq_port: int
    consecutive_misses: int = 0


@dataclass
class _P2PLookupJob:
    handle: PrefetchHandle
    """ The handle returned by the storage manager """

    keys: list[ObjectKey]
    """ The object keys submitted for this lookup, in request order """

    key_groups: list[GroupedObjectKeys]
    """ The same keys as submitted to the storage manager, one row per
    object group; the status result is one bitmap per row """


class P2PController:
    """Serves lookup requests from peers and maintains one L2 adapter per peer.

    P2P is enabled when ``p2p_config`` carries an advertise URL; otherwise the
    controller only answers lookup/unlock RPCs.

    Args:
        ctx: Shared engine context providing the storage manager and friends.
        p2p_config: Peer-to-peer configuration; inert when its advertise URL is
            empty.
        coordinator_config: Coordinator connection used for peer discovery.
        instance_id: Stable id of this instance, used to exclude itself from the
            discovered peer set.
        request_transport: Request transport exposed by this server and its
            discovered peers.
    """

    def __init__(
        self,
        ctx: MPCacheServerContext,
        p2p_config: P2PConfig,
        coordinator_config: CoordinatorConfig,
        instance_id: str,
        request_transport: Literal["zmq", "grpc"] = "zmq",
    ) -> None:
        self._ctx = ctx
        self._p2p_config = p2p_config
        self._instance_id = instance_id
        self._request_scheme = "grpc" if request_transport == "grpc" else "tcp"
        self._next_task_id = 0
        self._jobs: dict[int, _P2PLookupJob] = {}
        self._job_lock = threading.Lock()

        # Orchestration state (guarded by _orch_lock; written only by the poll
        # thread, read by report_status).
        self._orch_lock = threading.Lock()
        self._state = _P2PState.UNREGISTERED
        self._adapters: dict[str, _PeerAdapter] = {}

        self._instances_url = ""
        self._http_client: httpx.Client | None = None
        self._poll_thread: PeriodicThread | None = None

        if p2p_config.enabled:
            self._start_orchestration(coordinator_config)

        self._setup_metrics()

    def _start_orchestration(self, coordinator_config: CoordinatorConfig) -> None:
        """Initialize the transfer channel and start the peer-poll thread.

        Args:
            coordinator_config: Coordinator connection used for peer discovery.
        """
        initialize_transfer_channel_context(
            self._p2p_config.transfer_engine,
            self._ctx.storage_manager.l1_memory_desc,
            listen_url=self._p2p_config.effective_listen_url,
            advertise_url=self._p2p_config.advertise_url,
        )
        self._instances_url = coordinator_config.url.rstrip("/") + "/instances"
        timeout = max(1.0, coordinator_config.heartbeat_interval)
        self._http_client = httpx.Client(timeout=timeout)
        self._poll_thread = create_periodic_thread(
            name="p2p-controller-thread",
            interval=coordinator_config.heartbeat_interval,
            execute_fn=self._poll_cycle,
            level=ThreadLevel.MEDIUM,
        )
        self._poll_thread.start()

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict[str, object]:
        """Return module-specific status information.

        Returns:
            Dictionary with the active lookup-job count, the P2P registration
            state, and the set of connected peer instance ids.
        """
        with self._orch_lock:
            state = self._state.value
            peers = sorted(self._adapters.keys())
        return {
            "active_p2p_lookup_jobs": self._active_job_count(),
            "p2p_enabled": self._p2p_config.enabled,
            "p2p_state": state,
            "p2p_peer_count": len(peers),
            "p2p_peers": peers,
        }

    def close(self) -> None:
        """Stop the poll thread and release peer adapters and the channel."""
        if self._poll_thread is not None:
            self._poll_thread.stop()
            self._poll_thread = None

        for peer_id in list(self._adapters.keys()):
            self._remove_adapter(peer_id)

        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

        if self._p2p_config.enabled:
            delete_transfer_channel_context()

    # -----------------------------------------------------------------
    # RPC Handlers
    # -----------------------------------------------------------------

    @request_handler(RequestType.P2P_LOOKUP_AND_LOCK, HandlerType.BLOCKING)
    def p2p_lookup_and_lock(
        self,
        keys: list[ObjectKey],
        group_layout_descs: dict[int, MemoryLayoutDesc],
    ) -> int:
        """Submit a lookup and lock.

        Read-locks every L1-resident key (sparse; gaps allowed), not only
        the contiguous prefix.

        Args:
            keys: the list of object keys to look up and lock.
            group_layout_descs: memory layout of the objects, per object
                group.

        Returns:
            A unique task id (int) for querying the lookup status later
        """
        with self._job_lock:
            task_id = self._next_task_id
            self._next_task_id += 1

        key_groups = _group_keys_by_object_group(keys, group_layout_descs)
        if key_groups:
            # NOTE: skip_l2=True -- only objects already resident in L1 are
            # locked; "full" keeps every resident key, gaps allowed.
            handle = self._ctx.storage_manager.submit_prefetch_task(
                PrefetchTaskSpec(key_groups=key_groups, fetching_policy="full"),
                external_request_id=f"p2p-{task_id}",
                skip_l2=True,
            )
        else:
            # Nothing to look up: an already-complete empty handle.
            handle = PrefetchHandle(
                prefetch_request_id=-1,
                external_request_id=f"p2p-{task_id}",
                l1_found_indices=(),
                l1_hit_chunks=0,
                total_requested_keys=0,
                submit_time=time.monotonic(),
            )

        with self._job_lock:
            self._jobs[task_id] = _P2PLookupJob(
                handle=handle, keys=keys, key_groups=key_groups
            )

        logger.debug(
            "P2P lookup submitted: task_id=%d, %d keys, %d L1 hits",
            task_id,
            len(keys),
            len(handle.l1_found_indices),
        )
        return task_id

    @request_handler(RequestType.P2P_QUERY_LOOKUP_RESULTS, HandlerType.BLOCKING)
    def p2p_query_lookup_results(
        self,
        task_id: int,
    ) -> list[TransferChannelAddress] | None:
        """Query the results of the lookup request specified by the task ID.

        Returning a list of TransferChannelAddress objects indicates when the
        lookup is completed. None indicates the lookup has not completed yet.

        The returned list will always have the same length as the number of
        keys submitted in the corresponding p2p_lookup_and_lock call. For
        objects that is not found, the corresponding TransferChannelAddress
        will have an invalid offset (negative value).

        Args:
            task_id: The unique task ID returned by p2p_lookup_and_lock.

        Returns:
            A list of TransferChannelAddress objects if the lookup is complete,
            or None if the lookup is still in progress or the result has been
            queried. (Exactly once request)
        """
        with self._job_lock:
            job = self._jobs.get(task_id)
        if job is None:
            logger.warning(
                "P2P lookup job %d not found (already consumed or invalid)",
                task_id,
            )
            return None

        found_rows = self._ctx.storage_manager.query_prefetch_status(job.handle)
        if found_rows is None:
            # Still in progress (only possible once L2 prefetch is enabled).
            return None

        addresses = self._build_addresses(job, found_rows)

        with self._job_lock:
            self._jobs.pop(task_id, None)
        return addresses

    @request_handler(RequestType.P2P_UNLOCK_OBJECTS, HandlerType.BLOCKING)
    def p2p_unlock_objects(
        self,
        keys: list[ObjectKey],
    ) -> None:
        """Unlock the specified object keys.

        Args:
            keys: the list of object keys to unlock.
        """
        if not keys:
            return
        self._ctx.storage_manager.finish_read_prefetched(keys)

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _build_addresses(
        self,
        job: _P2PLookupJob,
        found_rows: list[Bitmap],
    ) -> list[TransferChannelAddress]:
        """Build the per-key transfer addresses for a completed lookup.

        ``found_rows`` is parallel to ``job.key_groups``; the keys at its set
        bits are the locked L1 hits, whose addresses are read via
        ``unsafe_read``. Every other key gets an invalid address. Addresses
        are returned in ``job.keys`` (request) order.
        """
        addresses = [_INVALID_ADDRESS] * len(job.keys)
        if not job.key_groups:
            return addresses
        found_keys = [
            key
            for row, found in zip(job.key_groups, found_rows, strict=True)
            for key in found.gather(row.keys)
        ]
        if not found_keys:
            return addresses

        good_keys, good_objs = self._ctx.storage_manager.unsafe_read(found_keys)
        obj_by_key = dict(zip(good_keys, good_objs, strict=True))

        positions: dict[ObjectKey, list[int]] = {}
        for i, key in enumerate(job.keys):
            positions.setdefault(key, []).append(i)
        for key in found_keys:
            obj = obj_by_key.get(key)
            if obj is None:
                # Locked but unreadable (e.g. evicted under a race); leave it
                # marked invalid so the peer skips it.
                continue
            address = TransferChannelAddress(
                offset=obj.shm_offset,
                size=obj.shm_byte_length,
            )
            for i in positions.get(key, ()):
                addresses[i] = address
        return addresses

    # -----------------------------------------------------------------
    # Orchestration (poll thread)
    # -----------------------------------------------------------------

    def _poll_cycle(self) -> ThreadRunSummary:
        """Run one discovery cycle: poll the coordinator and reconcile adapters.

        Returns:
            A summary of the cycle's resulting state and adapter changes.
        """
        try:
            instances = self._fetch_instances()
        except httpx.TimeoutException:
            return self._apply_state(_P2PState.DISCONNECTED, {})
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("P2P instance discovery failed: %s", e)
            return self._apply_state(_P2PState.UNREGISTERED, {})
        except Exception:
            logger.exception("Unexpected error during P2P instance discovery")
            return self._apply_state(_P2PState.UNREGISTERED, {})

        if not self._is_self_registered(instances):
            return self._apply_state(_P2PState.UNREGISTERED, {})

        upstream = {
            inst.instance_id: inst
            for inst in instances
            if inst.instance_id != self._instance_id
            and inst.p2p_advertised_url
            and inst.ip
            and inst.mq_port
        }
        return self._apply_state(_P2PState.REGISTERED, upstream)

    def _fetch_instances(self) -> list[_PeerInstance]:
        """Fetch and parse the coordinator's active-instance list.

        Returns:
            The parsed peer instances.

        Raises:
            httpx.HTTPError: If the request fails or returns a non-2xx status.
            ValueError: If the response body is not the expected shape.
        """
        assert self._http_client is not None
        response = self._http_client.get(self._instances_url)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("instances"), list):
            raise ValueError("malformed /instances response")
        instances: list[_PeerInstance] = []
        for raw in body["instances"]:
            instance_id = raw.get("instance_id", "")
            if not instance_id:
                continue
            instances.append(
                _PeerInstance(
                    instance_id=instance_id,
                    ip=raw.get("ip", ""),
                    p2p_advertised_url=raw.get("p2p_advertised_url", ""),
                    mq_port=int(raw.get("mq_port", 0) or 0),
                )
            )
        return instances

    def _is_self_registered(self, instances: list[_PeerInstance]) -> bool:
        """Return whether this instance appears in ``instances`` with our URL.

        Args:
            instances: The peer instances parsed from the coordinator.

        Returns:
            ``True`` if our instance id is present and advertises our URL.
        """
        return any(
            inst.instance_id == self._instance_id
            and inst.p2p_advertised_url == self._p2p_config.advertise_url
            for inst in instances
        )

    def _apply_state(
        self,
        state: _P2PState,
        upstream: dict[str, _PeerInstance],
    ) -> ThreadRunSummary:
        """Record the new state and reconcile adapters against ``upstream``.

        Args:
            state: The registration state derived from this cycle.
            upstream: Peer instances to maintain adapters for (empty unless
                registered).

        Returns:
            A summary describing the state and adapter changes this cycle.
        """
        with self._orch_lock:
            self._state = state
        added, removed = self._reconcile(upstream)
        return ThreadRunSummary(
            success=True,
            message=(
                f"state={state.value} peers={len(upstream)} "
                f"added={added} removed={removed}"
            ),
        )

    def _reconcile(self, upstream: dict[str, _PeerInstance]) -> tuple[int, int]:
        """Bring the local peer adapters in line with ``upstream``.

        Args:
            upstream: The peers that should each have a live adapter.

        Returns:
            A ``(added, removed)`` count of adapter changes this cycle.
        """
        added = 0
        removed = 0
        for peer_id, inst in upstream.items():
            current = self._adapters.get(peer_id)
            if current is None:
                if self._add_adapter(inst):
                    added += 1
            elif self._peer_changed(current, inst):
                self._remove_adapter(peer_id)
                removed += 1
                if self._add_adapter(inst):
                    added += 1
            else:
                current.consecutive_misses = 0

        for peer_id in list(self._adapters.keys()):
            if peer_id in upstream:
                continue
            adapter = self._adapters[peer_id]
            adapter.consecutive_misses += 1
            if adapter.consecutive_misses > _MAX_MISSES:
                self._remove_adapter(peer_id)
                removed += 1
        return added, removed

    @staticmethod
    def _peer_changed(adapter: _PeerAdapter, inst: _PeerInstance) -> bool:
        """Return whether a peer's advertised connection info changed.

        Args:
            adapter: The currently tracked adapter for the peer.
            inst: The peer's freshly discovered info.

        Returns:
            ``True`` if any connection field differs.
        """
        return (
            adapter.p2p_advertised_url != inst.p2p_advertised_url
            or adapter.ip != inst.ip
            or adapter.mq_port != inst.mq_port
        )

    def _add_adapter(self, inst: _PeerInstance) -> bool:
        """Create and register a peer L2 adapter for ``inst``.

        Args:
            inst: The peer to connect to.

        Returns:
            ``True`` if the adapter was created and tracked.
        """
        config = P2PL2AdapterConfig(
            peer_mq_server_url=(f"{self._request_scheme}://{inst.ip}:{inst.mq_port}"),
            peer_transfer_channel_server_url=inst.p2p_advertised_url,
            lookup_timeout_s=self._p2p_config.lookup_timeout,
            load_timeout_s=self._p2p_config.load_timeout,
        )
        try:
            adapter_id = self._ctx.storage_manager.add_l2_adapter(config)
        except Exception:
            logger.exception("Failed to add P2P adapter for peer %s", inst.instance_id)
            return False
        with self._orch_lock:
            self._adapters[inst.instance_id] = _PeerAdapter(
                adapter_id=adapter_id,
                ip=inst.ip,
                p2p_advertised_url=inst.p2p_advertised_url,
                mq_port=inst.mq_port,
            )
        logger.debug(
            "Added P2P adapter %d for peer %s (%s)",
            adapter_id,
            inst.instance_id,
            inst.p2p_advertised_url,
        )
        return True

    def _remove_adapter(self, peer_id: str) -> None:
        """Drain and remove the peer L2 adapter for ``peer_id``.

        Args:
            peer_id: Instance id of the peer whose adapter to remove.
        """
        with self._orch_lock:
            adapter = self._adapters.pop(peer_id, None)
        if adapter is None:
            return
        try:
            self._ctx.storage_manager.delete_l2_adapter(adapter.adapter_id)
        except Exception:
            logger.exception("Failed to remove P2P adapter for peer %s", peer_id)
            return
        logger.debug("Removed P2P adapter for peer %s", peer_id)

    def _active_job_count(self) -> int:
        """Return the number of active P2P lookup jobs (thread-safe)."""
        with self._job_lock:
            return len(self._jobs)

    def _setup_metrics(self) -> None:
        """Register OTel observable gauges for P2P controller metrics."""
        _gauge = partial(register_gauge, "lmcache.mp_server")
        _gauge(
            "lmcache_mp.active_p2p_lookup_jobs",
            "Number of active P2P lookup jobs",
            self._active_job_count,
        )
