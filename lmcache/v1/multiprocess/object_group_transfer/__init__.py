# SPDX-License-Identifier: Apache-2.0
"""Object-group KV transfer for the multiprocess server.

Moves chunks between LMCache memory objects and the engine's paged KV cache.
Each chunk is moved by the transfer engine its L1 object selects. See
``docs/design/v1/multiprocess/lmcache_driven_transfer_copy_paths.md``.
Import the public names from this package.
"""

# First Party
from lmcache.v1.multiprocess.object_group_transfer.direct_copy import (
    direct_transfer_supported,
    run_direct_transfer,
)
from lmcache.v1.multiprocess.object_group_transfer.dispatch import (
    transfer_kv_per_object_group,
)
from lmcache.v1.multiprocess.object_group_transfer.planning import (
    batched_iteration_with_skip,
    downsample_and_stage_block_ids,
    recalculate_blocks_to_skip,
)
from lmcache.v1.multiprocess.object_group_transfer.unfull_chunk import (
    get_unfull_chunk_layout_desc,
    get_unfull_chunk_staging_buffer,
    has_unfull_chunk_lead_block,
    split_unfull_chunk_block_ids,
    transfer_unfull_chunk,
    trim_and_stage_unfull_chunk_block_ids,
)

__all__ = [
    "batched_iteration_with_skip",
    "direct_transfer_supported",
    "downsample_and_stage_block_ids",
    "get_unfull_chunk_layout_desc",
    "get_unfull_chunk_staging_buffer",
    "has_unfull_chunk_lead_block",
    "recalculate_blocks_to_skip",
    "run_direct_transfer",
    "split_unfull_chunk_block_ids",
    "transfer_kv_per_object_group",
    "transfer_unfull_chunk",
    "trim_and_stage_unfull_chunk_block_ids",
]
