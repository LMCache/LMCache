# SPDX-License-Identifier: Apache-2.0
"""Node-local token-based L1 deletion.

:class:`DeleteService` resolves a token sequence to per-rank keys and deletes the
resident L1 objects, honoring (or, with ``force``, bypassing) L1 locks and pins,
and returns the resolved keys so the coordinator can pin-aware delete L2. L2 is
not touched here: the coordinator owns the L2 pin set and dispatches the L2
delete to this node's ``DELETE /cache/objects`` handler. Raises domain errors
(see :mod:`cache_control.errors`) that the HTTP layer maps to status codes.
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


class DeleteService:
    """Delete token sequences from L1 on one node.

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
        """Encode object keys to JSON-safe dicts (for the coordinator's L2 delete)."""
        return [asdict(key.to_encoded_object_key()) for key in obj_keys]

    def delete(
        self,
        model_name: str,
        world_size: int,
        token_ids: list[int],
        cache_salt: str,
        tier: Tier,
        force: bool,
    ) -> dict[str, object]:
        """Delete a token sequence's chunks from L1 (skipped when ``tier`` is ``l2``).

        Args:
            model_name: Model whose layout resolves the tokens to keys.
            world_size: World size selecting the layout and per-rank fan-out.
            token_ids: Prompt tokens whose complete chunks should be deleted.
            cache_salt: Per-tenant isolation salt applied to the produced keys.
            tier: ``l1`` / ``all`` delete L1; ``l2`` only resolves the keys.
            force: When True, delete L1 keys even if locked or pinned.

        Returns:
            ``{"requested", "deleted", "skipped", "resolved_keys", "status"}``.
            ``deleted``/``skipped`` are 0 for ``tier`` ``l2``; ``resolved_keys``
            (encoded) feed the coordinator's L2 delete; ``status`` is
            ``"deleted"`` or ``"noop"`` (sub-chunk sequence).

        Raises:
            InvalidRequest: token cap exceeded, or an invalid key field.
            Unavailable: no layout registered for the model.
        """
        obj_keys, chunks = self._resolve(model_name, world_size, token_ids, cache_salt)
        if not chunks:
            return {
                "requested": 0,
                "deleted": 0,
                "skipped": 0,
                "resolved_keys": [],
                "status": "noop",
            }
        deleted = skipped = 0
        if tier in (Tier.L1, Tier.ALL):
            deleted, skipped = self._engine.storage_manager.delete_l1_keys(
                obj_keys, force=force
            )
        return {
            "requested": chunks,
            "deleted": deleted,
            "skipped": skipped,
            "resolved_keys": self._encode_keys(obj_keys),
            "status": "deleted",
        }
