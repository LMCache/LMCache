# SPDX-License-Identifier: Apache-2.0
"""Object-group KV transfer for the multiprocess server.

Per-kernel-group gather and scatter between the engine's paged KV cache and
LMCache memory objects, driven by the cache context's ``KVLayerGroupsManager``:
block-id downsampling for sub-chunk sliding windows, skip recalculation, and
the per-object-group transfer plan the copy kernels run.

Two copy paths are available, chosen per object group by
:func:`direct_transfer_supported`: the direct path moves each chunk straight
between the pinned host object and the paged buffers with one
``cudaMemcpyBatchAsync`` per chunk, and is used whenever the build, the driver
and the object group's layout allow it; everything else stages each chunk
through a GPU temp buffer and scatters it with the block transfer kernel. See
``docs/design/v1/multiprocess/lmcache_driven_transfer_copy_paths.md``.

Modules: ``planning`` (batching, block-id downsampling, skip arithmetic),
``direct_copy`` and ``staged`` (the two copy paths), ``dispatch`` (picks one
per object group) and ``unfull_chunk``. Import the public names from this
package.
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
