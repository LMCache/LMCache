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

# Standard
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LMCKVCacheGroup:
    """One LMCache view of an engine-side KV cache group."""

    engine_kv_cache_group_id: int
    """Engine-side KV cache group ID used to choose the request block-id list."""

    layer_names: tuple[str, ...]
    """Layer names assigned to this engine KV cache group."""


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

    @property
    def num_engine_kv_cache_groups(self) -> int:
        """Number of engine block-id lists expected on each transfer request."""
        if not self.groups:
            return 1
        return max(group.engine_kv_cache_group_id for group in self.groups) + 1

    def to_layout_hints(
        self,
        registered_layer_names: Sequence[str],
    ) -> dict[str, object] | None:
        """Build layout hints for the current registered KV cache order."""
        if not self.groups or not registered_layer_names:
            return None

        layer_to_pos = {name: idx for idx, name in enumerate(registered_layer_names)}
        per_layer_engine_group_idx = [0] * len(layer_to_pos)
        matched_layers: set[str] = set()

        for group in self.groups:
            for layer_name in group.layer_names:
                pos = layer_to_pos.get(layer_name)
                if pos is not None:
                    per_layer_engine_group_idx[pos] = group.engine_kv_cache_group_id
                    matched_layers.add(layer_name)

        if matched_layers:
            missing_layers = set(layer_to_pos) - matched_layers
            if missing_layers:
                raise ValueError(
                    "Engine KV cache groups did not cover registered KV cache "
                    f"layers: {sorted(missing_layers)[:8]}"
                )
            return {"per_layer_engine_group_idx": per_layer_engine_group_idx}

        if self.num_engine_kv_cache_groups > 1:
            raise ValueError(
                "Unable to map registered KV cache layers to engine KV cache "
                "groups for HMA."
            )

        return None
