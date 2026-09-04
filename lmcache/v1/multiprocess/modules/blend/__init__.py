# SPDX-License-Identifier: Apache-2.0
"""Blend (context-blend / cross-request KV reuse) package for MPCacheServer.

The public surface is :class:`BlendModule`; the remaining re-exports keep
the pre-split ``modules.blend`` import surface working for the existing
tests until they are repointed at the submodules.
"""

# First Party
# Legacy re-export: tests gate on the native plan via this module attribute.
from lmcache import device_ops as device_ops
from lmcache.v1.mp_coordinator.blend_client import PENDING as PENDING
from lmcache.v1.multiprocess.modules.blend.lookup import _CBUnifiedJob as _CBUnifiedJob
from lmcache.v1.multiprocess.modules.blend.matcher import (
    BlendTokenRangeMatcher as BlendTokenRangeMatcher,
    _unique_token_coverage as _unique_token_coverage,
)
from lmcache.v1.multiprocess.modules.blend.module import BlendModule
from lmcache.v1.multiprocess.modules.blend.read_set import (
    _CBReadGroups as _CBReadGroups,
    _cb_chunk_major_object_keys as _cb_chunk_major_object_keys,
    _classify_cb_read_groups as _classify_cb_read_groups,
    _narrow_attn_desc as _narrow_attn_desc,
)
from lmcache.v1.multiprocess.modules.blend.retrieve import (
    _HAS_NATIVE_RETRIEVE_PLAN as _HAS_NATIVE_RETRIEVE_PLAN,
)
from lmcache.v1.multiprocess.modules.blend.rope import (
    _cb_group_rope_geometry as _cb_group_rope_geometry,
    _CBRopeState as _CBRopeState,
)
from lmcache.v1.multiprocess.modules.blend.scatter_fallback import (
    _group_slot_mappings as _group_slot_mappings,
)

__all__ = ["BlendModule"]
