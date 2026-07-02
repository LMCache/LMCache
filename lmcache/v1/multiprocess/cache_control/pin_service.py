# SPDX-License-Identifier: Apache-2.0
"""Node-local token-based L1 pin/unpin operations.

:class:`PinService` resolves a token sequence to per-rank keys and pins/unpins
the resident L1 objects. It also returns the resolved keys (encoded) so the
coordinator can pin/unpin the same keys in its L2 eviction. It raises
transport-agnostic domain errors (see :mod:`cache_control.errors`); the HTTP
layer maps those to status codes.
"""

# Standard
from dataclasses import asdict
from typing import Any

# First Party
from lmcache.v1.distributed.tiers import Tier
from lmcache.v1.multiprocess.cache_control.errors import InvalidRequest
from lmcache.v1.multiprocess.cache_control.key_resolver import (
    MAX_TOKEN_IDS,
    resolve_l1_keys,
)


class PinService:
    """Pin and unpin token sequences in L1 on one node.

    Args:
        engine: The node's cache engine (resolves tokens and holds L1).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def _resolve(
        self,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
    ) -> tuple[list, int]:
        """Validate inputs and resolve the token sequence to ``(obj_keys, chunks)``.

        Raises:
            InvalidRequest: token cap exceeded, or an invalid key field.
            Unavailable: no layout registered for the model.
        """
        if len(token_ids) > MAX_TOKEN_IDS:
            raise InvalidRequest(
                f"too many token_ids in a single request "
                f"(limit={MAX_TOKEN_IDS}, got={len(token_ids)})"
            )
        obj_keys, chunks, _ = resolve_l1_keys(
            self._engine, model_name, world_size, token_ids, cache_salt
        )
        return obj_keys, chunks

    @staticmethod
    def _encode_keys(obj_keys: list) -> list[dict[str, object]]:
        """Encode object keys to JSON-safe dicts (for the coordinator's L2 pin)."""
        return [asdict(key.to_encoded_object_key()) for key in obj_keys]

    def pin(
        self,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
        tier: Tier,
    ) -> dict[str, object]:
        """Pin a token sequence's chunks in L1 (skipped when ``tier`` is ``l2``).

        Returns:
            ``{"requested", "pinned", "resolved_keys", "status"}``; ``requested``
            is the chunk count, ``pinned`` the number of resident L1 keys pinned
            (0 when ``tier`` is ``l2``), ``resolved_keys`` the resolved keys
            (encoded, for the coordinator's L2 pin), and ``status`` ``"pinned"``
            or ``"noop"``.

        Raises:
            InvalidRequest: token cap exceeded, or an invalid key field.
            Unavailable: no layout registered for the model.
        """
        obj_keys, chunks = self._resolve(model_name, world_size, token_ids, cache_salt)
        if not chunks:
            return {"requested": 0, "pinned": 0, "resolved_keys": [], "status": "noop"}
        pinned = (
            self._engine.storage_manager.pin_l1_keys(obj_keys)
            if tier in (Tier.L1, Tier.ALL)
            else 0
        )
        return {
            "requested": chunks,
            "pinned": pinned,
            "resolved_keys": self._encode_keys(obj_keys),
            "status": "pinned",
        }

    def unpin(
        self,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
        tier: Tier,
    ) -> dict[str, object]:
        """Unpin a token sequence's chunks in L1 (skipped when ``tier`` is ``l2``).

        Returns:
            ``{"requested", "unpinned", "resolved_keys", "status"}``; ``requested``
            is the chunk count, ``unpinned`` the number of resident L1 pins
            released (0 when ``tier`` is ``l2``), ``resolved_keys`` the resolved
            keys (encoded, for the coordinator's L2 unpin), and ``status``
            ``"unpinned"`` or ``"noop"``.

        Raises:
            InvalidRequest: token cap exceeded, or an invalid key field.
            Unavailable: no layout registered for the model.
        """
        obj_keys, chunks = self._resolve(model_name, world_size, token_ids, cache_salt)
        if not chunks:
            return {
                "requested": 0,
                "unpinned": 0,
                "resolved_keys": [],
                "status": "noop",
            }
        unpinned = (
            self._engine.storage_manager.unpin_l1_keys(obj_keys)
            if tier in (Tier.L1, Tier.ALL)
            else 0
        )
        return {
            "requested": chunks,
            "unpinned": unpinned,
            "resolved_keys": self._encode_keys(obj_keys),
            "status": "unpinned",
        }
