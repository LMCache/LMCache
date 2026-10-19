# SPDX-License-Identifier: Apache-2.0
"""What an eviction controller reports, whichever controller it is.

Eviction metrics are named by role, not by class: any controller whose
``stats()`` returns :class:`EvictionStats` and that registers itself as a
metric source (``register_metric_source(self)``) gets the
``eviction.*``, ``quota.*`` and ``pins.*`` metrics. The built-in
``FleetEvictionController`` and a replacement shipped in another package
fill the same type, so swapping one for the other keeps every dashboard.

A replacement may add tiers and label values (a ``pinned`` /
``unreachable`` / ``stale`` skip reason, an ``owner`` delete target), never
rename them. See ``docs/design/v1/mp_coordinator/observability.md``.
"""

# Standard
from dataclasses import dataclass, field

# The tenant label of every tenant without an explicit quota: one series for
# all of them keeps the label's cardinality to the quota table's.
OTHER_TENANTS = "_other"


@dataclass
class EvictionTierStats:
    """Eviction on one tier.

    Counts cover this process; gauges are read when the snapshot is taken.
    Tenant-keyed values use the tenant's ``cache_salt`` when it has an
    explicit quota, and :data:`OTHER_TENANTS` for the rest together.

    Attributes:
        sweeps: Sweeps by outcome: ``idle`` (nothing to evict),
            ``planned`` (deletes sent) or ``no_target`` (no server to send
            them to).
        keys_selected: Keys picked for eviction, by tenant.
        bytes_selected: Bytes of those keys, by tenant.
        keys_skipped: Keys passed over, by reason: ``pinned``,
            ``unreachable`` (every copy on a server that left) or ``stale``
            (nothing holds it any more).
        delete_requests_sent: Delete requests by target (``any_server``
            for a shared pool, ``owner`` for the server holding the copy),
            then by result (``ok``, ``error``).
        delete_requests_skipped: Delete requests not sent, by reason:
            ``owner_left`` or ``no_server``.
        lru_keys_tracked: Keys in this tier's eviction LRU.
        tenants_over_watermark: Tenants due for eviction now.
        tenant_bytes_used: Bytes per tenant; :data:`OTHER_TENANTS` sums the
            tenants without an explicit quota.
        tenant_bytes_limit: Quota per tenant; :data:`OTHER_TENANTS` is the
            default each such tenant gets, absent while none is set.
    """

    sweeps: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(("idle", "planned", "no_target"), 0)
    )
    keys_selected: dict[str, int] = field(default_factory=dict)
    bytes_selected: dict[str, int] = field(default_factory=dict)
    keys_skipped: dict[str, int] = field(default_factory=dict)
    delete_requests_sent: dict[str, dict[str, int]] = field(default_factory=dict)
    delete_requests_skipped: dict[str, int] = field(default_factory=dict)
    lru_keys_tracked: int = 0
    tenants_over_watermark: int = 0
    tenant_bytes_used: dict[str, int] = field(default_factory=dict)
    tenant_bytes_limit: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvictionStats:
    """What an eviction controller has done, and the budgets it enforces.

    Attributes:
        tiers: Per-tier eviction, keyed by tier (``l1``, ``l2``).
        delete_requests_in_flight: Delete requests sent and not yet
            answered.
        keys_pinned: Keys pinned against eviction.
    """

    tiers: dict[str, EvictionTierStats]
    delete_requests_in_flight: int
    keys_pinned: int
