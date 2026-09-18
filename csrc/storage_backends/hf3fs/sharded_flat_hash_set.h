// SPDX-License-Identifier: Apache-2.0
//
// sharded_flat_hash_set.h - A concurrent hash set using sharded locks
//
// This implementation wraps absl::flat_hash_set with per-shard locking to
// provide thread-safe concurrent access for write-heavy / read-light workloads.
//
// Dependencies: absl::flat_hash_set (header-only)
// License: Apache-2.0
//
// Authors: Wenwen Chen <wenwen.chen@samsung.com>
// Usage:
//   #include "sharded_flat_hash_set.h"
//
//   sharded_flat_hash_set<std::string> set;
//   set.Insert("key");
//   if (set.Contains("key")) { ... }
//   set.Erase("key");

#pragma once

#include "absl/container/flat_hash_set.h"
#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

namespace concurrent_hash {

// Default shard count (power of 2 for fast modulo)
// Increased from 16 to 64 to reduce lock contention at high worker counts
// (e.g., 32+ workers in HF3FS native connector)
constexpr size_t kDefaultShards = 64;

// Per-shard storage with its own mutex
template <typename Key>
struct Shard {
  absl::flat_hash_set<Key> set;
  mutable std::mutex mu;
};

// Concurrent hash set with sharded locks
template <typename Key, typename Hash = std::hash<Key>>
class sharded_flat_hash_set {
 public:
  sharded_flat_hash_set(size_t num_shards = kDefaultShards)
      : shards_(num_shards), num_shards_(num_shards) {}

  // Insert a key (returns true if inserted, false if already exists)
  bool Insert(const Key& key) {
    size_t idx = Hash()(key) % num_shards_;
    std::lock_guard<std::mutex> lock(shards_[idx].mu);
    return shards_[idx].set.insert(key).second;
  }

  // Insert with move semantics
  bool Insert(Key&& key) {
    size_t idx = Hash()(key) % num_shards_;
    std::lock_guard<std::mutex> lock(shards_[idx].mu);
    return shards_[idx].set.insert(std::move(key)).second;
  }

  // Check if key exists (thread-safe, no lock on read path)
  bool Contains(const Key& key) const {
    size_t idx = Hash()(key) % num_shards_;
    std::lock_guard<std::mutex> lock(shards_[idx].mu);
    return shards_[idx].set.count(key) > 0;
  }

  // Erase a key (returns true if erased)
  bool Erase(const Key& key) {
    size_t idx = Hash()(key) % num_shards_;
    std::lock_guard<std::mutex> lock(shards_[idx].mu);
    return shards_[idx].set.erase(key) > 0;
  }

  // Erase with move semantics
  bool Erase(Key&& key) {
    size_t idx = Hash()(key) % num_shards_;
    std::lock_guard<std::mutex> lock(shards_[idx].mu);
    return shards_[idx].set.erase(key) > 0;
  }

  // Clear all shards
  void Clear() {
    for (size_t i = 0; i < num_shards_; ++i) {
      std::lock_guard<std::mutex> lock(shards_[i].mu);
      shards_[i].set.clear();
    }
  }

  // Get number of elements
  size_t size() const {
    size_t total = 0;
    for (size_t i = 0; i < num_shards_; ++i) {
      std::lock_guard<std::mutex> lock(shards_[i].mu);
      total += shards_[i].set.size();
    }
    return total;
  }

  // Check if empty
  bool empty() const { return size() == 0; }

 private:
  std::vector<Shard<Key>> shards_;
  size_t num_shards_;
};

}  // namespace concurrent_hash