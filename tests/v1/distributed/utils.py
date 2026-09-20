# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for distributed tests."""

# Standard
from typing import Any

# First Party
from lmcache.v1.distributed.api import (
    GroupedKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.platform import current_device_spec


def should_use_lazy_alloc() -> bool:
    """Return whether the current platform supports lazy L1 allocation."""
    return current_device_spec.is_pin_supported


def single_row_spec(
    keys: list[ObjectKey], layout_desc: MemoryLayoutDesc, **spec_kwargs: Any
) -> PrefetchTaskSpec:
    """A prefetch spec with one key row: object group 0, a single kv rank.

    ``spec_kwargs`` are forwarded to :class:`PrefetchTaskSpec` (e.g.
    ``num_kv_readers``, ``fetching_policy``, ``lock_mode``).
    """
    return PrefetchTaskSpec(
        key_groups=[GroupedKeys(keys=keys, object_group_id=0, layout_desc=layout_desc)],
        **spec_kwargs,
    )


def ranked_spec(
    keys_by_rank: list[list[ObjectKey]],
    layout_desc: MemoryLayoutDesc,
    **spec_kwargs: Any,
) -> PrefetchTaskSpec:
    """A single-object-group prefetch spec with one key row per kv rank.

    ``keys_by_rank[r]`` is the chunk-ordered key list of kv rank ``r``.
    ``spec_kwargs`` are forwarded to :class:`PrefetchTaskSpec`.
    """
    return PrefetchTaskSpec(
        key_groups=[
            GroupedKeys(keys=keys, object_group_id=0, layout_desc=layout_desc)
            for keys in keys_by_rank
        ],
        **spec_kwargs,
    )
