# SPDX-License-Identifier: Apache-2.0
"""Coordinator-side token-based pin/unpin dispatch to MP servers."""

# Future
from __future__ import annotations

# Standard
from typing import Any

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.mp_coordinator.registry import MPInstance

logger = init_logger(__name__)


class PinManager:
    """Submit token-based pin/unpin requests to MP servers."""

    async def submit_pin(
        self,
        target: MPInstance,
        http_client: httpx.AsyncClient,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
        tier: str,
    ) -> dict[str, Any]:
        """``POST /cache/pins`` to ``target`` and return its JSON reply.

        Args:
            target: The MP server holding the objects.
            http_client: Shared async client for outbound coordinator calls.
            model_name: Model whose layout the target uses to resolve keys.
            world_size: World size selecting the layout and per-rank fan-out.
            token_ids: Prompt tokens whose complete chunks should be pinned.
            cache_salt: Per-tenant isolation salt applied to the produced keys.
            tier: Which tier(s) to pin (``l1`` / ``l2`` / ``all``); the server
                uses it to decide whether to pin its L1.

        Returns:
            The server's reply, e.g.
            ``{"requested", "pinned", "resolved_keys", "status"}``.

        Raises:
            httpx.HTTPError: If the target is unreachable or returns non-2xx.
        """
        url = f"http://{target.ip}:{target.http_port}/cache/pins"
        body = self._body(model_name, world_size, token_ids, cache_salt, tier)
        resp = await http_client.post(url, json=body)
        resp.raise_for_status()
        logger.info(
            "Pin submitted to %s: %d tokens", target.instance_id, len(token_ids)
        )
        return resp.json()

    async def submit_unpin(
        self,
        target: MPInstance,
        http_client: httpx.AsyncClient,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
        tier: str,
    ) -> dict[str, Any]:
        """``DELETE /cache/pins`` on ``target`` and return its JSON reply.

        Args:
            target: The MP server holding the objects.
            http_client: Shared async client for outbound coordinator calls.
            model_name: Model whose layout the target uses to resolve keys.
            world_size: World size selecting the layout and per-rank fan-out.
            token_ids: Prompt tokens whose complete chunks should be unpinned.
            cache_salt: Per-tenant isolation salt applied to the produced keys.
            tier: Which tier(s) to unpin (``l1`` / ``l2`` / ``all``); the server
                uses it to decide whether to unpin its L1.

        Returns:
            The server's reply, e.g.
            ``{"requested", "unpinned", "resolved_keys", "status"}``.

        Raises:
            httpx.HTTPError: If the target is unreachable or returns non-2xx.
        """
        url = f"http://{target.ip}:{target.http_port}/cache/pins"
        body = self._body(model_name, world_size, token_ids, cache_salt, tier)
        # ``httpx.AsyncClient.delete`` doesn't accept ``json=``;
        # ``request("DELETE", ...)`` is the supported form.
        resp = await http_client.request("DELETE", url, json=body)
        resp.raise_for_status()
        logger.info(
            "Unpin submitted to %s: %d tokens", target.instance_id, len(token_ids)
        )
        return resp.json()

    @staticmethod
    def _body(
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
        tier: str,
    ) -> dict[str, Any]:
        """Build the shared MP-server request body for pin and unpin."""
        return {
            "model_name": model_name,
            "world_size": world_size,
            "token_ids": token_ids,
            "cache_salt": cache_salt,
            "tier": tier,
        }
