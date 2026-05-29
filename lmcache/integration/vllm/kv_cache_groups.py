# SPDX-License-Identifier: Apache-2.0
"""Convert vLLM KV cache group metadata into LMCache's neutral model."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.gpu_connector.utils import LayoutHints

# First Party
from lmcache.v1.kv_cache_groups import LMCKVCacheGroup, LMCKVCacheGroups


def _vllm_kv_cache_groups(kv_cache_config: Any) -> Sequence[Any]:
    """Return vLLM KV cache groups, or an empty sequence when unavailable.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` (or ``None`` before the engine
            has produced one).

    Returns:
        The ``kv_cache_groups`` sequence from the config, or an empty tuple when
        the config is ``None`` or exposes no groups.
    """
    if kv_cache_config is None:
        return ()
    return getattr(kv_cache_config, "kv_cache_groups", ()) or ()


def lmcache_kv_cache_groups_from_vllm(
    kv_cache_config: Any,
    registered_layer_names: Sequence[str] | None = None,
) -> LMCKVCacheGroups:
    """Convert vLLM's KV cache group interface to LMCache's neutral type.

    The vLLM fields used here are from the v1 KV cache interface:
    ``KVCacheConfig.kv_cache_groups`` and ``KVCacheGroupSpec.layer_names``.
    Keep additional vLLM-specific field reads in this module, then normalize
    them into ``LMCKVCacheGroup`` fields before crossing LMCache boundaries.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` (or ``None``). Each entry in its
            ``kv_cache_groups`` becomes one ``LMCKVCacheGroup`` whose
            ``engine_kv_cache_group_id`` is the group's enumeration index.
        registered_layer_names: Layer names in registration order, used to map
            each group's ``layer_names`` to registered tensor indices. When
            ``None``, ``layer_indices`` are left empty (the engine-group count
            is still derivable, which is all the scheduler side needs).

    Returns:
        An ``LMCKVCacheGroups`` with one group per vLLM KV cache group; empty
        when ``kv_cache_config`` is ``None`` or exposes no groups.
    """
    layer_to_idx = (
        {name: idx for idx, name in enumerate(registered_layer_names)}
        if registered_layer_names is not None
        else {}
    )

    return LMCKVCacheGroups.from_groups(
        LMCKVCacheGroup(
            engine_kv_cache_group_id=engine_kv_cache_group_id,
            layer_names=tuple(getattr(group, "layer_names", ())),
            layer_indices=tuple(
                layer_to_idx[name]
                for name in getattr(group, "layer_names", ())
                if name in layer_to_idx
            ),
        )
        for engine_kv_cache_group_id, group in enumerate(
            _vllm_kv_cache_groups(kv_cache_config)
        )
    )


def inflated_lmcache_kv_cache_groups_from_vllm(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
    layout_hints: "LayoutHints | None" = None,
) -> LMCKVCacheGroups:
    """Build inflated LMCache KV layer groups from vLLM metadata and tensors.

    Combines the engine KV cache group metadata with the physical tensor layout
    so that layers are split both by engine block-id space and by transfer-kernel
    identity (see :func:`lmcache.v1.kv_layer_groups.inflate_lmc_kv_cache_groups`).

    Args:
        kv_cache_config: vLLM ``KVCacheConfig`` describing the engine KV cache
            groups.
        kv_caches: Registered KV tensors keyed by layer name, in registration
            order. Keys provide the layer->index mapping; values are inspected
            for physical shape/dtype.
        layout_hints: Optional engine-provided layout hints forwarded to format
            detection (e.g. ``NHD``/``HND`` and compression metadata).

    Returns:
        The inflated ``LMCKVCacheGroups`` whose order is the protocol-visible
        LMCache group order used by store/retrieve block IDs.
    """
    # First Party
    from lmcache.utils import EngineType
    from lmcache.v1.gpu_connector.utils import normalize_kv_and_discover_format
    from lmcache.v1.kv_layer_groups import inflate_lmc_kv_cache_groups

    lmc_kv_cache_groups = lmcache_kv_cache_groups_from_vllm(
        kv_cache_config,
        tuple(kv_caches.keys()),
    )
    gpu_kv_format, normalized_kv_caches = normalize_kv_and_discover_format(
        list(kv_caches.values()),
        EngineType.VLLM,
        layout_hints=layout_hints,
    )
    return inflate_lmc_kv_cache_groups(
        normalized_kv_caches,
        gpu_kv_format,
        lmc_kv_cache_groups,
    )
