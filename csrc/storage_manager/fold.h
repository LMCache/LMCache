// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "bitmap.h"

namespace lmcache {

namespace storage_manager {

/**
 * @brief Fold/unfold over the ``group x chunk x kv_rank`` ranked layout.
 *
 * Computes the longest model-wide prefix that every object group can serve
 * under its own rule (full attention or a cross-chunk sliding window), and the
 * concrete keys each group must retain to serve it.
 *
 * The input ``found`` is group-major / chunk-major / rank-minor: bit
 * ``g * (num_chunks * num_ranks) + j * num_ranks + r`` is set iff chunk ``j``
 * of object group ``g`` is present for kv_rank ``r``. A chunk counts as present
 * for a group only when every kv_rank shard is present.
 *
 * @param found Presence bitmap of length
 *     ``group_windows.size() * num_chunks * num_ranks``.
 * @param num_chunks Number of LMCache chunks in the request.
 * @param num_ranks Number of kv_rank shards per chunk.
 * @param group_windows Per-object-group cross-chunk sliding-window size in
 *     chunks, in object-group order; ``<= 0`` means full attention.
 *
 * @return ``{hit_length, retain_mask}``: ``hit_length`` in chunks and a retain
 *     mask over the same ranked layout as ``found`` (all ranks of each retained
 *     ``(group, chunk)`` set).
 */
std::pair<size_t, Bitmap> fold_unfold_ranked(
    const Bitmap& found, size_t num_chunks, size_t num_ranks,
    const std::vector<int64_t>& group_windows);

}  // namespace storage_manager

}  // namespace lmcache
