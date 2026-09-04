# SPDX-License-Identifier: Apache-2.0
"""BlendModule: composition of the blend mixins + engine-module wiring."""

# Standard
from collections import OrderedDict
from queue import Queue
from typing import TYPE_CHECKING, Any
import threading
import weakref

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.blend_client import BlendCoordinatorClient

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    HandlerSpec,
    InstanceLivenessTarget,
    ThreadPoolType,
)
from lmcache.v1.multiprocess.modules.blend.lookup import (
    LookupMixin,
    _CBUnifiedJob,
)
from lmcache.v1.multiprocess.modules.blend.matcher import BlendTokenRangeMatcher
from lmcache.v1.multiprocess.modules.blend.registration import RegistrationMixin
from lmcache.v1.multiprocess.modules.blend.retrieve import (
    RetrieveMixin,
    _DeviceEvent,
)
from lmcache.v1.multiprocess.modules.blend.rope import _CBRopeState
from lmcache.v1.multiprocess.modules.blend.scatter_fallback import (
    ScatterFallbackMixin,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.protocol import RequestType
from lmcache.v1.multiprocess.protocols.blend import handshake_response
from lmcache.v1.multiprocess.session import Session

logger = init_logger(__name__)


class BlendModule(
    LookupMixin,
    RegistrationMixin,
    RetrieveMixin,
    ScatterFallbackMixin,
    InstanceLivenessTarget,
):
    """Paged-aware blend. Wraps LMCacheDrivenTransfer STORE to register
    fingerprints; serves CB rope/lookup/retrieve RPCs; reads cross-module
    GPU state via :class:`LMCacheDrivenTransferModule.cache_contexts`."""

    #: ``Session.extras`` key: ``{"read_locks": N, "per_hash": {hash: keys}}``
    #: — the sparse lookup's reservation (N read locks per key, per #4866),
    #: written by the sparse classify and consumed by exactly one
    #: ``extras.pop`` — the retrieve, or :meth:`_release_unretrieved_locks`.
    UNRETRIEVED_KEYS_EXTRA = "cb.unretrieved_read_locked_keys"

    def __init__(
        self,
        ctx: MPCacheServerContext,
        lmcache_driven_transfer: LMCacheDrivenTransferModule,
        coordinator: "BlendCoordinatorClient | None" = None,
        enable_segmented_prefix: bool = False,
        enable_dedup_content: bool = False,
    ):
        # Blend reads the attention group + connector-private aux group only;
        # resolved per registration via _classify_cb_read_groups.
        self._ctx = ctx
        self._transfer_module = lmcache_driven_transfer
        # Server config (--enable-segmented-prefix): retain the gapped prefix on
        # a mid-prefix L2 retrieve failure instead of truncating at the gap.
        self._segmented_prefix = enable_segmented_prefix
        # Optional bridge to the fleet-wide fingerprint directory. ``None`` =>
        # purely local matching (publish/query paths skipped).
        self._coordinator = coordinator

        self._token_range_matcher = BlendTokenRangeMatcher(
            ctx.chunk_size, dedup_content=enable_dedup_content
        )
        self._event_bus = ctx.event_bus
        self._cb_rope_state: dict[int, _CBRopeState] = {}

        # Backstop for locks the retrieve never consumed (see
        # UNRETRIEVED_KEYS_EXTRA / _release_unretrieved_locks).
        ctx.session_manager.destroy_listeners.append(self._release_unretrieved_locks)

        # vLLM may call retrieve twice per request (partial- then full-block
        # alloc): ranges already scattered, so the repeat call skips them.
        # Bounded LRU keyed by (request id, WORKER id) -- at TP>1 each worker
        # issues its own retrieve and scatters into its own KV buffers, so the
        # key must include the worker or later ranks skip work they never did.
        self._cb_applied_match_ranges: "OrderedDict[tuple[str, int | None], set[tuple[bytes, int, int, tuple]]]" = OrderedDict()  # noqa: E501

        # Request-invariant retrieve-plan specs per GPU context (entries die
        # with the context). The cached tuple holds the rope_state it was
        # resolved against so a re-registration invalidates by identity.
        self._cb_plan_invariants: "weakref.WeakKeyDictionary[Any, tuple]" = (
            weakref.WeakKeyDictionary()
        )
        # Persistent pinned + device slot-mapping staging per GPU context,
        # grown on demand (see _cb_slot_buffers).
        self._cb_slot_staging: "weakref.WeakKeyDictionary[Any, tuple]" = (
            weakref.WeakKeyDictionary()
        )
        # Completion event per context; the next retrieve host-syncs it before
        # reusing the shared slot/temp buffers (_build_cb_retrieve_plan_flat).
        self._cb_plan_done_events: "weakref.WeakKeyDictionary[Any, _DeviceEvent]" = (
            weakref.WeakKeyDictionary()
        )
        # Retrieve-owned stream per context: keeps retrieves out of the shared
        # stream's store gather/commit traffic.
        self._cb_retrieve_streams: "weakref.WeakKeyDictionary[Any, Any]" = (
            weakref.WeakKeyDictionary()
        )
        self._cb_retrieve_cupy_streams: "weakref.WeakKeyDictionary[Any, Any]" = (
            weakref.WeakKeyDictionary()
        )

        # Non-blocking cb_unified_lookup poll state (submit-once, poll-on-recall)
        # so the handler never holds a worker thread across the L2->L1 loads.
        self._cb_jobs: dict[str, _CBUnifiedJob] = {}
        self._cb_jobs_lock = threading.Lock()

        # Async fingerprint registration: store enqueues, worker drains.
        # (tokens_in_range, chunk_hashes, start_chunk_idx, position_offset,
        # request_id). The request_id is the enqueuing store's, so the
        # registration event is attributed to it rather than to the drainer.
        _FpJob = tuple[list[int], list[bytes], int, int, str]
        self._fingerprint_queue: "Queue[_FpJob]" = Queue()
        self._fingerprint_stop = threading.Event()
        self._fingerprint_worker = threading.Thread(
            target=self._drain_fingerprint_queue,
            name="cb-fingerprint-worker",
            daemon=True,
        )
        self._fingerprint_worker.start()

        # In-flight fingerprint hashes; storage_gate keeps these from eviction.
        self._pending_fp_hashes: set[bytes] = set()
        self._pending_fp_lock = threading.Lock()

        # Lazy eviction strikes; evict only at threshold so async re-store
        # can refresh the bucket first.
        self._stale_strike: dict[bytes, int] = {}
        self._STALE_STRIKE_THRESHOLD = 2

    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        # STORE shadows LMCacheDrivenTransfer's; the compositor registers the
        # blend module last so this handler wins.
        return [
            HandlerSpec(RequestType.STORE, self.store, ThreadPoolType.AFFINITY),
            HandlerSpec(
                RequestType.CB_REGISTER_ROPE,
                self.cb_register_rope,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNREGISTER_ROPE,
                self.cb_unregister_rope,
                ThreadPoolType.SYNC,
            ),
            HandlerSpec(
                RequestType.CB_UNIFIED_LOOKUP,
                self.cb_unified_lookup,
                ThreadPoolType.NORMAL,
            ),
            HandlerSpec(
                RequestType.CB_RETRIEVE_PRE_COMPUTED,
                self.cb_retrieve_pre_computed,
                ThreadPoolType.AFFINITY,
            ),
            HandlerSpec(
                RequestType.CB_PROTOCOL_HANDSHAKE,
                self.cb_protocol_handshake,
                ThreadPoolType.SYNC,
            ),
        ]

    def cb_protocol_handshake(self, client_version: int) -> tuple[int, bool]:
        return handshake_response(client_version)

    def report_status(self) -> dict:
        # Meta is derived live from MP server gpu_transfe

        cache_contexts = self._transfer_module.context_entries_snapshot()

        def _meta(iid: int) -> "tuple[str, int] | None":
            entry = cache_contexts.get(iid)
            return (entry.model_name, entry.world_size) if entry is not None else None

        return {
            "registered_cb_rope_instances": list(self._cb_rope_state.keys()),
            "cb_rope_meta": {str(iid): _meta(iid) for iid in self._cb_rope_state},
            "active_cb_lookups": len(self._cb_jobs),
        }

    def _release_unretrieved_locks(self, session: Session) -> None:
        """Release read locks the request's retrieve never consumed.

        A client whose matches are all covered by its own prefix cache sends
        no ``CB_RETRIEVE_PRE_COMPUTED``, so the retrieve's orphan sweep never
        runs and the sparse-prefetch locks would pin those chunks in L1 for
        the server's lifetime. Session destruction is the safe release point:
        no retrieve for the request can arrive afterwards.

        Args:
            session: The session being destroyed.
        """
        stash = session.extras.pop(self.UNRETRIEVED_KEYS_EXTRA, None)
        if not stash:
            return
        keys = [key for hash_keys in stash["per_hash"].values() for key in hash_keys]
        # Whole-reservation role: no retrieve consumed anything, so the full
        # reservation (N locks per key, per #4866) is still held.
        self._ctx.storage_manager.finish_read_prefetched(
            keys, read_locks=stash["read_locks"]
        )
        logger.info(
            "Released %d unretrieved read lock(s) for ended request %s",
            len(keys),
            session.request_id,
        )

    def close(self) -> None:
        listeners = self._ctx.session_manager.destroy_listeners
        if self._release_unretrieved_locks in listeners:
            listeners.remove(self._release_unretrieved_locks)
        self._fingerprint_stop.set()
        if self._coordinator is not None:
            # Joins the client's daemon thread and closes its httpx.Client;
            # otherwise the coordinator leg leaks both on server shutdown.
            self._coordinator.close()
        self._cb_rope_state.clear()

    def drop_instance_state(self, instance_id: int) -> None:
        """Drop blend state for a reaped instance (InstanceLivenessTarget hook).

        Only the CB rope state is held per instance; the GPU cache context is
        owned by ``LMCacheDrivenTransferModule`` (no mirror here), so reaping
        the GPU entry frees it directly. A no-op if no rope state is held.

        Args:
            instance_id: The reaped worker's instance ID.
        """
        if self._cb_rope_state.pop(instance_id, None) is not None:
            logger.info("Dropped CB rope state for reaped instance %d", instance_id)
