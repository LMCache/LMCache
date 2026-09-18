// SPDX-License-Identifier: Apache-2.0

#include "fold.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace lmcache {

namespace lmcache_native {

Bitmap fold(const Bitmap& found, size_t num_chunks, size_t num_ranks,
            const std::vector<int64_t>& group_windows) {
  const size_t num_groups = group_windows.size();
  // Chunk-major / group / rank-minor: bit
  // ``j * (num_groups * num_ranks) + g * num_ranks + r``.
  const size_t chunk_stride = num_groups * num_ranks;

  // ``servable[j]`` (bit ``j``, prefix length ``j + 1``) stays set only if
  // every group can serve a length-``j + 1`` prefix under its rule. ``run`` is
  // the count of consecutive present chunks ending at the current chunk, so a
  // length-L prefix needs the last ``min(window, L)`` chunks present, i.e.
  // ``run >= min(window, L)``.
  std::vector<char> servable(num_chunks, 1);
  for (size_t g = 0; g < num_groups; ++g) {
    const int64_t window = group_windows[g];
    const size_t eff_window =
        (window <= 0) ? num_chunks : static_cast<size_t>(window);
    const size_t gbase = g * num_ranks;
    size_t run = 0;
    for (size_t prefix_len = 1; prefix_len <= num_chunks; ++prefix_len) {
      const size_t cbase = (prefix_len - 1) * chunk_stride + gbase;
      bool chunk_present = true;
      for (size_t r = 0; r < num_ranks; ++r) {
        if (!found.test(cbase + r)) {
          chunk_present = false;
          break;
        }
      }
      run = chunk_present ? run + 1 : 0;
      if (servable[prefix_len - 1] && run < std::min(eff_window, prefix_len)) {
        servable[prefix_len - 1] = 0;
      }
    }
  }

  Bitmap servable_lengths(num_chunks);
  for (size_t j = 0; j < num_chunks; ++j) {
    if (servable[j]) servable_lengths.set(j);
  }
  return servable_lengths;
}

Bitmap unfold(size_t hit_length, size_t num_chunks, size_t num_ranks,
              const std::vector<int64_t>& group_windows) {
  if (hit_length > num_chunks) hit_length = num_chunks;
  const size_t num_groups = group_windows.size();
  // Chunk-major / group / rank-minor: bit
  // ``j * (num_groups * num_ranks) + g * num_ranks + r``.
  const size_t chunk_stride = num_groups * num_ranks;

  // The chunks each group needs to serve ``hit_length``, expanded over every
  // kv_rank. In chunk-major layout a group's retained cells are strided across
  // chunks, so fill each retained ``(chunk, group)`` cell (its ``num_ranks``
  // bits) with a ``set_range``.
  Bitmap retain_mask(num_chunks * chunk_stride);
  if (hit_length == 0) return retain_mask;
  for (size_t g = 0; g < num_groups; ++g) {
    const int64_t window = group_windows[g];
    size_t lo = 0;
    if (window > 0 && hit_length > static_cast<size_t>(window)) {
      lo = hit_length - static_cast<size_t>(window);
    }
    const size_t gbase = g * num_ranks;
    for (size_t j = lo; j < hit_length; ++j) {
      const size_t base = j * chunk_stride + gbase;
      retain_mask.set_range(base, base + num_ranks);
    }
  }
  return retain_mask;
}

Bitmap fold_grouped(const std::vector<Bitmap>& rows, size_t num_ranks,
                    const std::vector<int64_t>& group_windows) {
  const size_t num_groups = group_windows.size();
  if (rows.size() != num_groups * num_ranks) {
    throw std::invalid_argument(
        "fold_grouped: expected " + std::to_string(num_groups * num_ranks) +
        " rows (groups x ranks), got " + std::to_string(rows.size()));
  }
  const size_t num_chunks = rows.empty() ? 0 : rows[0].size();
  for (const Bitmap& row : rows) {
    if (row.size() != num_chunks) {
      throw std::invalid_argument(
          "fold_grouped: all rows must have the same size, got " +
          std::to_string(row.size()) + " vs " + std::to_string(num_chunks));
    }
  }

  // ``servable[j]`` (prefix length ``j + 1``) stays set only if every group
  // can serve that length: ``run`` counts consecutive present chunks, and a
  // length-L prefix needs ``run >= min(window, L)``. Chunk ``j`` is present
  // for group ``g`` iff every rank row ``rows[g * num_ranks + r]`` has bit j.
  std::vector<char> servable(num_chunks, 1);
  for (size_t g = 0; g < num_groups; ++g) {
    const int64_t window = group_windows[g];
    const size_t eff_window =
        (window <= 0) ? num_chunks : static_cast<size_t>(window);
    const Bitmap* group_rows = rows.data() + g * num_ranks;
    size_t run = 0;
    for (size_t prefix_len = 1; prefix_len <= num_chunks; ++prefix_len) {
      const size_t j = prefix_len - 1;
      bool chunk_present = true;
      for (size_t r = 0; r < num_ranks; ++r) {
        if (!group_rows[r].test(j)) {
          chunk_present = false;
          break;
        }
      }
      run = chunk_present ? run + 1 : 0;
      if (servable[j] && run < std::min(eff_window, prefix_len)) {
        servable[j] = 0;
      }
    }
  }

  Bitmap servable_lengths(num_chunks);
  for (size_t j = 0; j < num_chunks; ++j) {
    if (servable[j]) servable_lengths.set(j);
  }
  return servable_lengths;
}

std::vector<Bitmap> unfold_grouped(size_t hit_length, size_t num_chunks,
                                   size_t num_ranks,
                                   const std::vector<int64_t>& group_windows) {
  if (hit_length > num_chunks) hit_length = num_chunks;
  const size_t num_groups = group_windows.size();

  std::vector<Bitmap> rows;
  rows.reserve(num_groups * num_ranks);
  for (size_t g = 0; g < num_groups; ++g) {
    const int64_t window = group_windows[g];
    size_t lo = 0;
    if (window > 0 && hit_length > static_cast<size_t>(window)) {
      lo = hit_length - static_cast<size_t>(window);
    }
    for (size_t r = 0; r < num_ranks; ++r) {
      Bitmap row(num_chunks);
      if (hit_length > 0) row.set_range(lo, hit_length);
      rows.push_back(std::move(row));
    }
  }
  return rows;
}

}  // namespace lmcache_native

}  // namespace lmcache
