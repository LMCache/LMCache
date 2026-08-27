# SPDX-License-Identifier: Apache-2.0
"""Memory-usage endpoints for the ``/instances`` collection.

Joins bytes used (``CacheUsageManager``) against capacity declared
(``ServerConfigRegistry``). Read-only: never evicts, throttles, or pushes.
A ``null`` ratio means capacity is undeclared, not that the compartment is
empty.

``GET /instances/usage`` is a literal path under a collection that also
takes ``{instance_id}`` segments. Routers are discovered alphabetically, so
``instances_api`` registers first -- harmless while it declares no
``GET /instances/{instance_id}``, and guarded by a test that would fail if
one ever shadowed this route.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request, status

# First Party
from lmcache.v1.distributed.api import ModuleMemoryCapacity, Tier
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    FleetMemoryResponse,
    InstanceMemoryStatus,
    ModuleMemoryStatus,
)
from lmcache.v1.mp_coordinator.views.server_config import (
    UNDECLARED_CAPACITY,
    ServerConfigRegistry,
)
from lmcache.v1.mp_coordinator.views.usage_manager import CacheUsageManager

router = APIRouter()

# ``CacheUsageManager`` keys fleet-shared pools under this instance id.
# Their bytes belong to the fleet: counted once, never per mount.
_SHARED_OWNER = ""

_TIERS = (Tier.L1, Tier.L2)


def _usage_by_owner(
    usage_manager: CacheUsageManager,
) -> dict[str, dict[tuple[Tier, str], int]]:
    """Collect used bytes per owner across both tiers.

    Returns:
        ``instance_id`` -> ``(tier, backend)`` -> bytes. Shared pools appear
        under :data:`_SHARED_OWNER`.
    """
    by_owner: dict[str, dict[tuple[Tier, str], int]] = {}
    for tier in _TIERS:
        for owner, backends in usage_manager.get_bytes_by_instance(tier).items():
            for backend, used in backends.items():
                by_owner.setdefault(owner, {})[(tier, backend)] = used
    return by_owner


def _to_status(
    tier: Tier, backend: str, used_bytes: int, capacity_bytes: int, shared: bool
) -> ModuleMemoryStatus:
    """Join one compartment's usage to its declared capacity.

    Returns:
        The joined status. ``usage_ratio`` is ``None`` when there is no
        capacity to divide by.
    """
    return ModuleMemoryStatus(
        tier=tier,
        backend=backend,
        shared=shared,
        used_bytes=used_bytes,
        capacity_bytes=capacity_bytes,
        usage_ratio=(
            used_bytes / capacity_bytes
            if capacity_bytes > UNDECLARED_CAPACITY
            else None
        ),
    )


def _instance_status(
    instance_id: str,
    used: dict[tuple[Tier, str], int],
    declared: tuple[ModuleMemoryCapacity, ...],
    registered: bool,
) -> InstanceMemoryStatus:
    """Build one server's status from its usage and its declaration.

    Declared-but-empty compartments report ``used_bytes=0``, so a freshly
    started server does not look unmonitored.

    Returns:
        The assembled status, compartments sorted by ``(tier, backend)``.
    """
    capacities = {(m.tier, m.backend): m.capacity_bytes for m in declared}
    statuses = [
        _to_status(
            tier,
            backend,
            used_bytes,
            capacities.get((tier, backend), UNDECLARED_CAPACITY),
            shared=False,
        )
        for (tier, backend), used_bytes in used.items()
    ]
    for module in declared:
        if module.shared or (module.tier, module.backend) in used:
            continue
        statuses.append(
            _to_status(
                module.tier, module.backend, 0, module.capacity_bytes, shared=False
            )
        )
    statuses.sort(key=lambda m: (m.tier.value, m.backend))
    return InstanceMemoryStatus(
        instance_id=instance_id,
        registered=registered,
        declared_capacity=bool(declared),
        modules=statuses,
    )


def _shared_capacities(
    declarations: dict[str, tuple[ModuleMemoryCapacity, ...]],
) -> dict[tuple[Tier, str], int]:
    """Resolve each shared pool's capacity across its declaring servers.

    One pool is one store, so declarations should agree. Disagreement reads
    as undeclared -- picking one would make the answer depend on
    registration order.

    Returns:
        ``(tier, backend)`` -> agreed capacity, else
        :data:`UNDECLARED_CAPACITY`.
    """
    claims: dict[tuple[Tier, str], set[int]] = {}
    for modules in declarations.values():
        for module in modules:
            if module.shared:
                claims.setdefault((module.tier, module.backend), set()).add(
                    module.capacity_bytes
                )
    return {
        identity: values.pop() if len(values) == 1 else UNDECLARED_CAPACITY
        for identity, values in claims.items()
    }


@router.get("/instances/usage")
async def fleet_usage(request: Request) -> FleetMemoryResponse:
    """Return the memory status of every MP server and shared pool.

    Args:
        request: Carries the coordinator context.

    Returns:
        A :class:`FleetMemoryResponse`. A server appears if it is
        registered, still holds bytes, or declared capacity -- so one that
        deregistered with L2 placements surviving is not dropped.
    """
    ctx = get_context(request)
    declarations = ctx.views.get(ServerConfigRegistry).get_all()
    registered = {instance.instance_id for instance in ctx.registry.all_instances()}
    by_owner = _usage_by_owner(ctx.views.get(CacheUsageManager))
    owned = {owner for owner in by_owner if owner != _SHARED_OWNER}

    instances = [
        _instance_status(
            instance_id=instance_id,
            used=by_owner.get(instance_id, {}),
            declared=declarations.get(instance_id, ()),
            registered=instance_id in registered,
        )
        for instance_id in sorted(registered | owned | set(declarations))
    ]

    shared_caps = _shared_capacities(declarations)
    shared = sorted(
        (
            _to_status(
                tier,
                backend,
                used_bytes,
                shared_caps.get((tier, backend), UNDECLARED_CAPACITY),
                shared=True,
            )
            for (tier, backend), used_bytes in by_owner.get(_SHARED_OWNER, {}).items()
        ),
        key=lambda m: (m.tier.value, m.backend),
    )
    return FleetMemoryResponse(instances=instances, shared_modules=shared)


@router.get("/instances/{instance_id}/usage")
async def instance_usage(instance_id: str, request: Request) -> InstanceMemoryStatus:
    """Return one MP server's memory status.

    Args:
        instance_id: The server to report on.
        request: Carries the coordinator context.

    Returns:
        That server's :class:`InstanceMemoryStatus`.

    Raises:
        HTTPException: 404 when the id is unknown -- not registered, holding
            no bytes, and having declared nothing.
    """
    ctx = get_context(request)
    declared = ctx.views.get(ServerConfigRegistry).get(instance_id)
    used = _usage_by_owner(ctx.views.get(CacheUsageManager)).get(instance_id, {})
    registered = ctx.registry.contains(instance_id)
    if not registered and not declared and not used:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown instance {instance_id!r}",
        )
    return _instance_status(
        instance_id=instance_id,
        used=used,
        declared=declared,
        registered=registered,
    )
