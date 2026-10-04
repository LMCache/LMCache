// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "dax_coordinated_l1/layout.h"

namespace lmcache::dax_coordinated_l1 {

bool try_acquire_peterson(BucketPetersonRecord* record,
                          std::uint32_t participant_id);
void release_peterson(BucketPetersonRecord* record,
                      std::uint32_t participant_id);

// Heap-ordered binary tree of participant_count - 1 two-way Peterson nodes.
// A caller retains each lower node until it releases all higher nodes.
bool try_acquire_tournament(BucketPetersonRecord* nodes,
                            std::uint32_t participant_id,
                            std::uint32_t participant_count);
void release_tournament(BucketPetersonRecord* nodes,
                        std::uint32_t participant_id,
                        std::uint32_t participant_count);

}  // namespace lmcache::dax_coordinated_l1
