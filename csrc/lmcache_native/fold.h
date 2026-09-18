// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "bitmap.h"

namespace lmcache {

namespace lmcache_native {

/**
 * @brief Fold per-(group, chunk, rank) presence into servable prefix lengths.
 *
 * For each object group, computes which prefix lengths it can serve under its
 * rule (full attention or a cross-chunk sliding window), and intersects across
 * groups. The result feeds :func:`Bitmap::highest_set_bit`; the model-wide hit
 * length is that index plus one (``-1`` -> hit length 0), then :func:`unfold`.
 *
 * The input ``found`` is chunk-major / group / rank-minor: bit
 * ``j * (num_groups * num_ranks) + g * num_ranks + r`` is set iff chunk ``j``
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
 * @return A bitmap of size ``num_chunks``; bit ``j`` set iff every group can
 *     serve a length-``j + 1`` prefix.
 */
Bitmap fold(const Bitmap& found, size_t num_chunks, size_t num_ranks,
            const std::vector<int64_t>& group_windows);

/**
 * @brief Expand a model-wide hit length into the per-group retain mask.
 *
 * Each group retains the chunks it needs to serve ``hit_length``: ``[0,
 * hit_length)`` for full attention, ``[hit_length - window, hit_length)`` for a
 * sliding window. The mask is over the same ranked layout as :func:`fold`'s
 * input (all kv_ranks of each retained ``(group, chunk)`` set).
 *
 * @param hit_length Model-wide prefix hit length in chunks (clamped to
 *     ``num_chunks``).
 * @param num_chunks Number of LMCache chunks in the request.
 * @param num_ranks Number of kv_rank shards per chunk.
 * @param group_windows Per-object-group cross-chunk sliding-window size in
 *     chunks, in object-group order; ``<= 0`` means full attention.
 *
 * @return Retain mask of length
 *     ``group_windows.size() * num_chunks * num_ranks``.
 */
Bitmap unfold(size_t hit_length, size_t num_chunks, size_t num_ranks,
              const std::vector<int64_t>& group_windows);

/**
 * @brief Fold per-(object group, kv_rank) row presence into servable prefix
 * lengths.
 *
 * The presence is given as one bitmap per ``(object group, kv_rank)`` row:
 * ``rows[g * num_ranks + r]`` is the presence of object group ``g`` on
 * kv_rank ``r``, and its bit ``j`` is set iff chunk ``j`` is present. A chunk
 * counts as present for a group only when every one of its rank rows has the
 * bit set. Every row has the same size, which is the number of chunks.
 *
 * @param rows Group-major / rank-minor row bitmaps,
 *     ``group_windows.size() * num_ranks`` of them, all of equal size.
 * @param num_ranks Number of kv_rank shards per object group.
 * @param group_windows Per-object-group cross-chunk sliding-window size in
 *     chunks, in object-group order; ``<= 0`` means full attention.
 *
 * @return A bitmap of size ``num_chunks``; bit ``j`` set iff every group can
 *     serve a length-``j + 1`` prefix.
 *
 * @throws std::invalid_argument If the row count is not
 *     ``group_windows.size() * num_ranks`` or the rows differ in size.
 */
Bitmap fold_grouped(const std::vector<Bitmap>& rows, size_t num_ranks,
                    const std::vector<int64_t>& group_windows);

/**
 * @brief Expand a model-wide hit length into per-(object group, kv_rank)
 * retain bitmaps.
 *
 * Each group retains the chunks it needs to serve ``hit_length``: ``[0,
 * hit_length)`` for full attention, ``[hit_length - window, hit_length)`` for a
 * sliding window. The mask is returned as one bitmap per row, group-major /
 * rank-minor: ``result[g * num_ranks + r]`` has size ``num_chunks`` and bit
 * ``j`` set iff object group ``g`` must retain chunk ``j`` (every rank of a
 * group gets the same mask).
 *
 * @param hit_length Model-wide prefix hit length in chunks (clamped to
 *     ``num_chunks``).
 * @param num_chunks Number of LMCache chunks in the request.
 * @param num_ranks Number of kv_rank shards per object group.
 * @param group_windows Per-object-group cross-chunk sliding-window size in
 *     chunks, in object-group order; ``<= 0`` means full attention.
 *
 * @return ``group_windows.size() * num_ranks`` retain bitmaps of size
 *     ``num_chunks``, group-major / rank-minor.
 */
std::vector<Bitmap> unfold_grouped(size_t hit_length, size_t num_chunks,
                                   size_t num_ranks,
                                   const std::vector<int64_t>& group_windows);

}  // namespace lmcache_native

}  // namespace lmcache
