# SPDX-License-Identifier: Apache-2.0
"""LMCache-owned KV cache group metadata.

Engine integrations should convert their native KV cache specs into these
neutral LMCache types before sending metadata over multiprocess IPC. This file
is the intended home for engine-derived fields LMCache may need to normalize in
the future, such as sliding-window windows, Mamba state groups, or logical vs.
physical block-size details.

Do not put vLLM/SGLang-specific object access here. Engine-specific conversion
belongs in the corresponding ``lmcache.integration.<engine>`` package.
"""

# Future
from __future__ import annotations

# Standard
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LMCKVCacheGroup:
    """One LMCache view of an engine-side KV cache group."""

    engine_kv_cache_group_id: int
    """Engine-side KV cache group ID used to choose the request block-id list."""

    layer_names: tuple[str, ...]
    """Layer names assigned to this engine KV cache group."""

    layer_indices: tuple[int, ...] = ()
    """Registered KV tensor indices assigned to this engine KV cache group."""

    def to_serializable(self) -> list[Any]:
        return [
            self.engine_kv_cache_group_id,
            list(self.layer_names),
            list(self.layer_indices),
        ]

    @classmethod
    def from_serializable(cls, payload: list[Any]) -> "LMCKVCacheGroup":
        if len(payload) != 3:
            raise ValueError(f"Invalid LMCKVCacheGroup payload: {payload}")
        engine_kv_cache_group_id, layer_names, layer_indices = payload
        return cls(
            engine_kv_cache_group_id=int(engine_kv_cache_group_id),
            layer_names=tuple(str(name) for name in layer_names),
            layer_indices=tuple(int(idx) for idx in layer_indices),
        )


@dataclass(frozen=True)
class LMCKVCacheGroups:
    """LMCache's neutral representation of engine-side KV cache groups."""

    groups: tuple[LMCKVCacheGroup, ...] = ()

    @classmethod
    def from_groups(cls, groups: Iterable[LMCKVCacheGroup]) -> "LMCKVCacheGroups":
        groups_tuple = tuple(groups)
        bad_ids = [
            group.engine_kv_cache_group_id
            for group in groups_tuple
            if group.engine_kv_cache_group_id < 0
        ]
        if bad_ids:
            raise ValueError(
                f"engine_kv_cache_group_id must be non-negative: {bad_ids}"
            )
        return cls(groups_tuple)

    def serialize(self) -> str:
        """Serialize to a stable JSON string for ZMQ transport."""
        payload = {
            "version": 1,
            "groups": [group.to_serializable() for group in self.groups],
        }
        return json.dumps(payload, separators=(",", ":"))

    @classmethod
    def deserialize(cls, serialized: str | None) -> "LMCKVCacheGroups":
        """Deserialize the JSON string produced by :meth:`serialize`."""
        if not serialized:
            return cls()
        payload = json.loads(serialized)
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported LMCKVCacheGroups payload: {payload}")
        return cls.from_groups(
            LMCKVCacheGroup.from_serializable(group)
            for group in payload.get("groups", [])
        )

    @property
    def num_engine_kv_cache_groups(self) -> int:
        """Number of engine block-id lists expected on each transfer request."""
        if not self.groups:
            return 1
        return max(group.engine_kv_cache_group_id for group in self.groups) + 1

    @property
    def num_lmc_kv_cache_groups(self) -> int:
        """Number of LMCache KV layer groups visible to transfer requests."""
        if not self.groups:
            return 1
        return len(self.groups)

    def engine_group_ids_by_lmc_group(self) -> tuple[int, ...]:
        """Return the engine KV cache group ID for each LMCache group."""
        if not self.groups:
            return (0,)
        return tuple(group.engine_kv_cache_group_id for group in self.groups)

    def expand_engine_block_ids_to_lmc_groups(
        self,
        block_ids_per_engine_group: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """Expand engine-group block IDs to LMCache-group block IDs.

        vLLM reports block IDs indexed by its engine KV cache groups. LMCache
        transfer requests are indexed by inflated LMCache KV layer groups, so
        each LMCache group reuses the block IDs from its source engine group.
        """
        if not block_ids_per_engine_group:
            return [[] for _ in self.engine_group_ids_by_lmc_group()]

        block_ids_per_lmc_group: list[list[int]] = []
        for engine_group_id in self.engine_group_ids_by_lmc_group():
            if engine_group_id >= len(block_ids_per_engine_group):
                raise ValueError(
                    "Missing block IDs for engine KV cache group "
                    f"{engine_group_id}; got {len(block_ids_per_engine_group)} "
                    "engine groups"
                )
            block_ids_per_lmc_group.append(
                list(block_ids_per_engine_group[engine_group_id])
            )
        return block_ids_per_lmc_group

    def per_layer_engine_group_indices(
        self,
        num_registered_layers: int,
    ) -> list[int] | None:
        """Return engine group index per registered KV tensor."""
        if not self.groups or num_registered_layers == 0:
            return None

        per_layer_engine_group_idx = [0] * num_registered_layers
        matched_indices: set[int] = set()

        for group in self.groups:
            for layer_idx in group.layer_indices:
                if layer_idx < 0 or layer_idx >= num_registered_layers:
                    raise ValueError(
                        f"Layer index {layer_idx} is outside registered layer "
                        f"range [0, {num_registered_layers})"
                    )
                per_layer_engine_group_idx[layer_idx] = group.engine_kv_cache_group_id
                matched_indices.add(layer_idx)

        if matched_indices:
            missing_indices = set(range(num_registered_layers)) - matched_indices
            if missing_indices:
                raise ValueError(
                    "Engine KV cache groups did not cover registered KV "
                    f"cache layer indices: {sorted(missing_indices)[:8]}"
                )
            return per_layer_engine_group_idx

        if self.num_engine_kv_cache_groups > 1:
            raise ValueError(
                "Unable to map registered KV cache tensors to engine KV cache "
                "groups for HMA."
            )

        return None
