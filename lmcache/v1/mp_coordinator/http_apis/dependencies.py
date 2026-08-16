# SPDX-License-Identifier: Apache-2.0
"""Typed per-app context for the coordinator's HTTP handlers.

Instead of each handler reaching into ``request.app.state`` with stringly-typed
``getattr`` calls, handlers resolve a single :class:`CoordinatorContext` via
:func:`get_context`. The context is built complete in ``create_app``. The
outbound HTTP client is deliberately *not* part of it: it must bind to the
running event loop, so the lifespan parks it on ``app.state`` and dispatch
handlers fetch it via :func:`get_outbound_client`.
"""

# Standard
from dataclasses import dataclass

# Third Party
from fastapi import Request
import httpx

# First Party
from lmcache.v1.mp_coordinator.controllers.eviction_controller import (
    FleetEvictionController,
)
from lmcache.v1.mp_coordinator.controllers.prefetch_manager import PrefetchManager
from lmcache.v1.mp_coordinator.controllers.usage_manager import CacheUsageManager
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.ingest.http_event_source import HttpCacheEventSource
from lmcache.v1.mp_coordinator.key_directory import KeyDirectory
from lmcache.v1.mp_coordinator.persistence.metadata import MetadataPersister
from lmcache.v1.mp_coordinator.registry import InstanceRegistry
from lmcache.v1.mp_coordinator.server_config import ServerConfigRegistry
from lmcache.v1.multiprocess.token_hasher import TokenHasher


@dataclass
class CoordinatorContext:
    """Shared collaborators the coordinator's HTTP handlers operate on.

    Attributes:
        registry: Fleet membership (``MPInstance`` by ``instance_id``).
        usage_manager: Fleet byte usage per tier, rolled up by
            ``cache_salt`` and by reporting instance.
        eviction_controller: The fleet L2 eviction control loop. Owns the
            quota registry (``.quota``) and the L2 pin set, and enforces
            the former against the ``l2`` half of ``usage_manager``.
        prefetch_manager: Warm-prefetch proxy to MP servers.
        token_hasher: Resolves a pin request's ``token_ids`` to object keys
            (configured to match the fleet's ``chunk_size`` / ``hash_algorithm``).
        key_directory: Fleet-wide key → placements directory built from
            MP-server cache events (eventually consistent).
        event_gate: Admission authority after a source delivers cache events.
        event_source: HTTP source adapter used by ``POST /events`` before
            batches reach the gate.
        metadata_persister: Durable store for operator intent. Every
            handler that changes a pin or a quota must ``save`` so the
            change survives a restart.
        server_config: Declared module capacities per MP server; the
            denominator for a usage ratio. Populated by ``config`` cache
            events.
    """

    registry: InstanceRegistry
    usage_manager: CacheUsageManager
    eviction_controller: FleetEvictionController
    prefetch_manager: PrefetchManager
    token_hasher: TokenHasher
    key_directory: KeyDirectory
    event_gate: EventGate
    event_source: HttpCacheEventSource
    metadata_persister: MetadataPersister
    server_config: ServerConfigRegistry


def get_context(request: Request) -> CoordinatorContext:
    """Return the per-app :class:`CoordinatorContext`.

    Args:
        request: The FastAPI request whose ``app.state`` carries the context.

    Returns:
        The shared :class:`CoordinatorContext`.

    Raises:
        RuntimeError: If the context is not initialized (wired by
            ``create_app``, so this should not happen in practice).
    """
    ctx = getattr(request.app.state, "ctx", None)
    if ctx is None:
        raise RuntimeError("coordinator context not initialized")
    return ctx


def get_outbound_client(request: Request) -> httpx.AsyncClient:
    """Return the lifespan-bound outbound client, or raise if unset.

    Args:
        request: The FastAPI request.

    Returns:
        The shared outbound :class:`httpx.AsyncClient`.

    Raises:
        RuntimeError: If accessed before the lifespan filled it in (e.g. a bare
            app with no startup).
    """
    client = getattr(request.app.state, "outbound_client", None)
    if client is None:
        raise RuntimeError("outbound client not initialized")
    return client
