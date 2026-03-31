// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — types.cpp
//
// Implements ipc_key_to_object_keys(), mirroring the Python version
// in lmcache/v1/distributed/api.py.

#include "types.h"

namespace lmcache {
namespace server {

std::vector<ObjectKey> ipc_key_to_object_keys(
    const std::string& model_name, int world_size, int worker_id,
    const std::vector<std::vector<uint8_t>>& chunk_hashes) {
  std::vector<ObjectKey> keys;
  keys.reserve(chunk_hashes.size() *
               static_cast<size_t>(worker_id < 0 ? world_size : 1));

  for (const auto& chunk_hash : chunk_hashes) {
    if (worker_id < 0) {
      // Lookup mode: expand each hash to all workers in world_size
      for (int wid = 0; wid < world_size; ++wid) {
        int32_t kv_rank =
            ObjectKey::compute_kv_rank(world_size, wid, world_size, wid);
        keys.push_back(ObjectKey{chunk_hash, model_name, kv_rank});
      }
    } else {
      int32_t kv_rank = ObjectKey::compute_kv_rank(world_size, worker_id,
                                                   world_size, worker_id);
      keys.push_back(ObjectKey{chunk_hash, model_name, kv_rank});
    }
  }

  return keys;
}

}  // namespace server
}  // namespace lmcache
