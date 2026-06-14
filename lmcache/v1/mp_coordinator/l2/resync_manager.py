# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side L2 resync.

On a fresh coordinator (or after a restart), the in-memory usage and
eviction trackers know nothing about keys already resident in L2. This
manager paginates an MP server's ``GET /l2/keys`` and feeds each entry
into :class:`L2UsageManager` + :class:`L2EvictionManager`, so quota
enforcement and LRU eviction work from a representative baseline
rather than from zero.

Best-effort: resync failures are logged and the manager gives up. The
ongoing usage-event stream from MP servers will eventually correct any
initial blind spots.

Wired into the coordinator app's lifespan as a one-shot background
task — see ``app.py``. Disable via ``enable_startup_resync=False``.
"""

# Future
from __future__ import annotations

# Standard
import asyncio

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import EncodedObjectKey
from lmcache.v1.mp_coordinator.l2.eviction_manager import L2EvictionManager
from lmcache.v1.mp_coordinator.l2.usage_manager import L2UsageManager
from lmcache.v1.mp_coordinator.registry import InstanceRegistry, MPInstance

logger = init_logger(__name__)


class L2ResyncManager:
    """Backfill the coordinator's L2 trackers from a live MP server's
    actual L2 contents.

    Args:
        usage_manager: The shared usage manager. Each resynced key
            contributes ``size_bytes`` under its ``cache_salt`` bucket.
        eviction_manager: The shared eviction manager. Each resynced
            key is registered via ``on_store(key, size_bytes)``.
        page_size: ``page_size`` query param forwarded to the MP
            server's ``/l2/keys`` endpoint. The server clamps this to
            its own ceiling.

    Note:
        The recorded state is a snapshot taken at the moment the
        listing was issued. Concurrent stores and evictions during a
        paginated walk may cause individual keys to be missed or
        double-counted — the contract is best-effort, not
        snapshot-isolated. See the design doc for
        ``l2_apis`` for the underlying listing semantics.
    """

    def __init__(
        self,
        usage_manager: L2UsageManager,
        eviction_manager: L2EvictionManager,
        page_size: int = 1000,
    ) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive (got {page_size})")
        self._usage_manager = usage_manager
        self._eviction_manager = eviction_manager
        self._page_size = page_size

    async def resync_from(
        self,
        instance: MPInstance,
        http_client: httpx.AsyncClient,
        request_timeout: float = 30.0,
    ) -> int:
        """Page through ``instance``'s L2 keys and record each one.

        Args:
            instance: The MP server to query.
            http_client: Shared async HTTP client.
            request_timeout: Per-page HTTP timeout in seconds.

        Returns:
            Total number of keys successfully recorded. Stops early on
            HTTP failure and returns whatever was recorded so far.
        """
        url = f"http://{instance.ip}:{instance.http_port}/l2/keys"
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
            pages += 1
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
                    obj_key = encoded.to_object_key()
                    size_bytes = int(entry["size_bytes"])
                except (KeyError, TypeError, ValueError) as exc:
                    logger.debug("Skipping unparsable resync entry %r: %s", entry, exc)
                    continue
                self._usage_manager.record_stored(obj_key, size_bytes)
                self._eviction_manager.on_store(obj_key)
                total += 1
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

        Args:
            registry: Live MP server registry.
            http_client: Shared async HTTP client.
            poll_interval: Seconds between registry checks.
            max_wait: Maximum seconds to wait for the first
                registration. After this the resync gives up and
                returns ``0``.
            request_timeout: Per-page HTTP timeout once a server is
                found.

        Returns:
            Total keys recorded, or ``0`` if no MP server registered
            within ``max_wait``.

        Note:
            Uses ``asyncio.get_running_loop().time()`` for the wait
            budget so timing is monotonic and unaffected by wall-clock
            adjustments.
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
