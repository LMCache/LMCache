// SPDX-License-Identifier: Apache-2.0

#include "dax_coordinated_l1/peterson.h"

#include "dax_coordinated_l1/visibility_x86.h"

namespace lmcache::dax_coordinated_l1 {

bool try_acquire_peterson(BucketPetersonRecord* record,
                          std::uint32_t participant_id) {
  if (participant_id > 1) {
    return false;
  }
  const auto other = 1U - participant_id;
  record->participant_flag[participant_id].value = 1;
  publish_range(&record->participant_flag[participant_id], kCacheLineBytes);
  record->victim.value = participant_id;
  publish_range(&record->victim, kCacheLineBytes);
  refresh_range(&record->participant_flag[other], kCacheLineBytes);
  refresh_range(&record->victim, kCacheLineBytes);
  if (record->participant_flag[other].value != 0 &&
      record->victim.value == participant_id) {
    record->participant_flag[participant_id].value = 0;
    publish_range(&record->participant_flag[participant_id], kCacheLineBytes);
    return false;
  }
  return true;
}

void release_peterson(BucketPetersonRecord* record,
                      std::uint32_t participant_id) {
  record->participant_flag[participant_id].value = 0;
  publish_range(&record->participant_flag[participant_id], kCacheLineBytes);
}

bool try_acquire_tournament(BucketPetersonRecord* nodes,
                            std::uint32_t participant_id,
                            std::uint32_t participant_count) {
  if ((participant_count != 2 && participant_count != 4) ||
      participant_id >= participant_count) {
    return false;
  }
  std::array<std::uint32_t, 2> path{};
  std::array<std::uint32_t, 2> sides{};
  std::size_t acquired = 0;
  auto child = participant_count - 1 + participant_id;
  while (child != 0) {
    const auto parent = (child - 1) / 2;
    const auto side = (child - 1) % 2;
    if (!try_acquire_peterson(nodes + parent, side)) {
      // The failed node cleared its own flag. Undo lower-node claims in
      // reverse acquisition order before the caller retries the bucket.
      while (acquired != 0) {
        --acquired;
        release_peterson(nodes + path[acquired], sides[acquired]);
      }
      return false;
    }
    path[acquired] = parent;
    sides[acquired++] = side;
    child = parent;
  }
  return true;
}

void release_tournament(BucketPetersonRecord* nodes,
                        std::uint32_t participant_id,
                        std::uint32_t participant_count) {
  // Release root first: otherwise a sibling could reach the same root-side
  // flag while this participant still owns it.
  if (participant_count == 4) {
    release_peterson(nodes, participant_id / 2);
    release_peterson(nodes + 1 + participant_id / 2, participant_id % 2);
  } else {
    release_peterson(nodes, participant_id);
  }
}

}  // namespace lmcache::dax_coordinated_l1
