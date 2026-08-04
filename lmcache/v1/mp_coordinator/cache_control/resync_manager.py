# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side L2 resync.

On boot, paginates an MP server's ``GET /cache/objects`` (the default
adapter, as before), synthesizes ``STORE`` batches, and feeds them to
the key directory (placements) and
— through the ``CacheEventBroadcaster`` — to every registered consumer (the
usage view, the eviction LRU), so quota enforcement starts from a
representative baseline rather than zero. The directory side goes
through :meth:`KeyDirectory.reconcile` — no stream-cursor bookkeeping —
so a backfill cannot disturb, or be rejected by, the instance's live
event stream. Best-effort: failures are logged and the manager gives
up; the ongoing event stream corrects any initial blind spots.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import asyncio

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import EncodedObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.mp_coordinator.cache_control.event_broadcaster import (
        CacheEventBroadcaster,
    )
    from lmcache.v1.mp_coordinator.key_directory import KeyDirectory
    from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstance

logger = init_logger(__name__)


class L2ResyncManager:
    """Backfill the coordinator's L2 trackers from a live MP server's
    actual L2 contents. Best-effort; not snapshot-isolated.

    Args:
        key_directory: The fleet key directory; backfilled via its
            gate-bypassing :meth:`KeyDirectory.reconcile`.
        event_broadcaster: The applied-batch fan-out; every synthesized
            batch is routed to the registered consumers.
        page_size: ``page_size`` forwarded to the MP server's
            ``/cache/objects`` endpoint.
    """

    def __init__(
        self,
        key_directory: KeyDirectory,
        event_broadcaster: CacheEventBroadcaster,
        page_size: int = 1000,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive (got {page_size})")
        self._key_directory = key_directory
        self._event_broadcaster = event_broadcaster
        self._page_size = page_size

    async def resync_from(
        self,
        instance: MPInstance,
        http_client: httpx.AsyncClient,
        request_timeout: float = 30.0,
    ) -> int:
        """Page through ``instance``'s default-adapter L2 keys.

        Returns the number of keys recorded; stops early with the
        partial count on HTTP failure or on a page missing ``adapter`` /
        ``shared`` (guessing the placement identity would leave
        placements no ``DELETE`` event can remove).
        """
        url = f"http://{instance.ip}:{instance.http_port}/cache/objects"
        page_token: str | None = None
        total = 0
        pages = 0
        while True:
            params: dict[str, str | int] = {"page_size": self._page_size}
            if page_token is not None:
                params["page_token"] = page_token
            try:
                resp = await http_client.get(
                    url, params=params, timeout=request_timeout
                )
                resp.raise_for_status()
                body = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning(
                    "Resync from %s failed at page %d (recorded %d so far): %s",
                    instance.instance_id,
                    pages,
                    total,
                    exc,
                )
                return total
            if not body.get("adapter") or "shared" not in body:
                logger.warning(
                    "Resync from %s: malformed page %d (adapter=%r, "
                    "shared present=%s); stopping (recorded %d so far)",
                    instance.instance_id,
                    pages,
                    body.get("adapter"),
                    "shared" in body,
                    total,
                )
                return total
            pages += 1
            batch_entries: list[CacheEventEntry] = []
            for entry in body.get("entries", []):
                try:
                    key = entry["key"]
                    encoded = EncodedObjectKey(
                        chunk_hash_hex=key["chunk_hash_hex"],
                        model_name=key["model_name"],
                        kv_rank=key["kv_rank"],
                        object_group_id=key.get("object_group_id", 0),
                        cache_salt=key.get("cache_salt", ""),
                    )
                    encoded.to_object_key()  # validates key invariants
                    size_bytes = int(entry["size_bytes"])
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug("Skipping unparsable resync entry %r: %s", entry, exc)
                    continue
                batch_entries.append(
                    CacheEventEntry(key=encoded, size_bytes=size_bytes)
                )
                total += 1
            if batch_entries:
                # Placement identity must match the live event stream's
                # (backend = adapter type name, plus the shared flag) so
                # later DELETE events remove the backfilled placements.
                batch = CacheEventBatch(
                    instance_id=instance.instance_id,
                    incarnation=0,
                    seq=pages,
                    event_type=CacheEventType.STORE,
                    tier=Tier.L2,
                    backend=str(body["adapter"]),
                    shared=bool(body["shared"]),
                    entries=batch_entries,
                )
                self._key_directory.reconcile(batch)
                self._event_broadcaster.broadcast(batch)
            page_token = body.get("next_page_token")
            if page_token is None:
                break
        logger.info(
            "Resync from %s complete: %d keys across %d page(s)",
            instance.instance_id,
            total,
            pages,
        )
        return total

    async def wait_and_resync(
        self,
        registry: InstanceRegistry,
        http_client: httpx.AsyncClient,
        poll_interval: float,
        max_wait: float,
        request_timeout: float = 30.0,
    ) -> int:
        """Poll the registry until an MP server registers, then resync.

        Returns the number of keys recorded; ``0`` if no MP server
        registered within ``max_wait``.
        """
        deadline = asyncio.get_running_loop().time() + max_wait
        while True:
            target = registry.random_instance()
            if target is not None:
                logger.info(
                    "Starting L2 resync from %s (%s:%d)",
                    target.instance_id,
                    target.ip,
                    target.http_port,
                )
                return await self.resync_from(
                    target, http_client, request_timeout=request_timeout
                )
            if asyncio.get_running_loop().time() >= deadline:
                logger.warning(
                    "L2 resync giving up: no MP servers registered within %ds",
                    int(max_wait),
                )
                return 0
            await asyncio.sleep(poll_interval)
