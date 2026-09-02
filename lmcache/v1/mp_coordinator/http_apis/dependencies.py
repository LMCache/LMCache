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
from typing import TypeVar

# Third Party
from fastapi import HTTPException, Request
import httpx

# First Party
from lmcache.v1.mp_coordinator.controllers.base import Controller
from lmcache.v1.mp_coordinator.discovery import Registry
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.persistence.metadata import MetadataPersister
from lmcache.v1.mp_coordinator.views.base import View
from lmcache.v1.multiprocess.token_hasher import TokenHasher

ControllerT = TypeVar("ControllerT", bound=Controller)


@dataclass
class CoordinatorContext:
    """Shared collaborators the coordinator's HTTP handlers operate on.

    Attributes:
        controllers: The coordinator's controllers, addressed by type --
            ``controllers.get(FleetEvictionController)`` for the fleet L2
            eviction loop (which owns the quota registry and the L2 pin
            set, enforced against the ``l2`` half of the usage view), and
            ``PrefetchManager`` for warm prefetch. Addressing by type only
            works for a class the coordinator imports, so a controller
            shipped out of tree registers its own routes instead (see
            ``HttpRoutes``) rather than being resolved here.
        token_hasher: Resolves a pin request's ``token_ids`` to object keys
            (configured to match the fleet's ``chunk_size`` / ``hash_algorithm``).
        views: The fleet's read models, addressed by type --
            ``views.get(KeyDirectory)`` for key → placements,
            ``CacheUsageManager`` for per-tier byte usage,
            ``InstanceRegistry`` for fleet membership, and
            ``ServerConfigRegistry`` for the declared capacities a usage
            ratio divides by.
        event_gate: Ingest entry point for the fleet cache-event stream
            (``POST /events``).
        metadata_persister: Durable store for operator intent. Every
            handler that changes a pin or a quota must ``save`` so the
            change survives a restart.
    """

    controllers: Registry[Controller]
    token_hasher: TokenHasher
    views: Registry[View]
    event_gate: EventGate
    metadata_persister: MetadataPersister


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


def require_controller(
    request: Request, controller_type: type[ControllerT]
) -> ControllerT:
    """Return a controller, or fail the request if it is disabled.

    An operator can leave a built-in controller unbuilt (see
    ``extra_config["disabled_controllers"]``), usually because something
    out of tree has taken over its job. The endpoints that speak for it
    then have nothing to act on, and say so rather than raising the
    ``KeyError`` that reaching for an unbuilt collaborator would.

    Args:
        request: The FastAPI request carrying the coordinator context.
        controller_type: The controller the endpoint acts on.

    Returns:
        The controller.

    Raises:
        HTTPException: 404 if it was not built.
    """
    controller = get_context(request).controllers.find(controller_type)
    if controller is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{controller_type.__name__} is disabled on this coordinator; "
                f"its endpoints are unavailable"
            ),
        )
    return controller


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
