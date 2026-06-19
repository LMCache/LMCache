// SPDX-License-Identifier: Apache-2.0

#include "fold.h"

#include <algorithm>

namespace lmcache {

namespace storage_manager {

std::pair<size_t, Bitmap> fold_unfold_ranked(
    const Bitmap& found, size_t num_chunks, size_t num_ranks,
    const std::vector<int64_t>& group_windows) {
  const size_t num_groups = group_windows.size();
  const size_t group_stride = num_chunks * num_ranks;
  const size_t num_keys = num_groups * group_stride;

  // Fold: ``servable[L]`` stays true only if every group can serve a length-L
  // prefix under its rule. ``run`` is the count of consecutive present chunks
  // ending at the current chunk, so a length-L prefix needs the last
  // ``min(window, L)`` chunks present, i.e. ``run >= min(window, L)``. Length 0
  // is always servable.
  std::vector<char> servable(num_chunks + 1, 1);
  for (size_t g = 0; g < num_groups; ++g) {
    const int64_t window = group_windows[g];
    const size_t eff_window =
        (window <= 0) ? num_chunks : static_cast<size_t>(window);
    const size_t gbase = g * group_stride;
    size_t run = 0;
    for (size_t prefix_len = 1; prefix_len <= num_chunks; ++prefix_len) {
      const size_t cbase = gbase + (prefix_len - 1) * num_ranks;
      bool chunk_present = true;
      for (size_t r = 0; r < num_ranks; ++r) {
        if (!found.test(cbase + r)) {
          chunk_present = false;
          break;
        }
      }
      run = chunk_present ? run + 1 : 0;
      if (servable[prefix_len] && run < std::min(eff_window, prefix_len)) {
        servable[prefix_len] = 0;
      }
    }
  }

  // Right-most servable prefix length (length 0 is always servable).
  size_t hit_length = 0;
  for (size_t prefix_len = num_chunks + 1; prefix_len-- > 0;) {
    if (servable[prefix_len]) {
      hit_length = prefix_len;
      break;
    }
  }

  // Unfold: the concrete chunks each group needs to serve ``hit_length``,
  // expanded over every kv_rank.
  Bitmap retain_mask(num_keys);
  if (hit_length > 0) {
    std::vector<size_t> indices;
    for (size_t g = 0; g < num_groups; ++g) {
      const int64_t window = group_windows[g];
      size_t lo = 0;
      if (window > 0 && hit_length > static_cast<size_t>(window)) {
        lo = hit_length - static_cast<size_t>(window);
      }
      const size_t gbase = g * group_stride;
      for (size_t j = lo; j < hit_length; ++j) {
        const size_t cbase = gbase + j * num_ranks;
        for (size_t r = 0; r < num_ranks; ++r) {
          indices.push_back(cbase + r);
        }
      }
    }
    retain_mask.batched_set(indices);
  }
  return {hit_length, retain_mask};
}

}  // namespace storage_manager

}  // namespace lmcache
