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
 * @brief Fold per-row presence bitmaps into servable prefix lengths.
 *
 * ``rows[i]`` and ``windows[i]`` describe one object: bit ``j`` of ``rows[i]``
 * is set iff chunk ``j`` of that object is present, and ``windows[i]`` is the
 * object's cross-chunk sliding-window size in chunks (``<= 0`` means full
 * attention). A prefix of length ``L`` is servable iff every row can serve it
 * under its own window, i.e. its last ``min(window, L)`` chunks are present.
 * No ordering or grouping of the rows is assumed. Every row has the same
 * size, which is the number of chunks.
 *
 * @param rows Presence bitmaps, all of equal size.
 * @param windows Per-row window sizes, parallel to ``rows``.
 *
 * @return A bitmap of size ``num_chunks``; bit ``j`` set iff every row can
 *     serve a length-``j + 1`` prefix.
 *
 * @throws std::invalid_argument If ``rows`` and ``windows`` differ in length
 *     or the rows differ in size.
 */
Bitmap fold_grouped(const std::vector<Bitmap>& rows,
                    const std::vector<int64_t>& windows);

/**
 * @brief Expand a model-wide hit length into per-row retain bitmaps.
 *
 * Row ``i`` retains the chunks it needs to serve ``hit_length`` under
 * ``windows[i]``: ``[0, hit_length)`` for full attention (``<= 0``),
 * ``[hit_length - window, hit_length)`` for a sliding window.
 *
 * @param hit_length Model-wide prefix hit length in chunks (clamped to
 *     ``num_chunks``).
 * @param num_chunks Number of LMCache chunks in the request.
 * @param windows Per-row cross-chunk sliding-window sizes in chunks.
 *
 * @return ``windows.size()`` retain bitmaps of size ``num_chunks``, parallel
 *     to ``windows``.
 */
std::vector<Bitmap> unfold_grouped(size_t hit_length, size_t num_chunks,
                                   const std::vector<int64_t>& windows);

}  // namespace lmcache_native

}  // namespace lmcache
