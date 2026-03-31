// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — L1 slab store implementation
//
// mmap'd slab allocator with per-slot state machine and LRU eviction.

#include "l1_store.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <sys/mman.h>  // mmap, munmap, MAP_HUGETLB

#include <cuda_runtime.h>  // cudaHostRegister, cudaHostUnregister

#include "storage_manager/ttl_lock.h"

namespace lmcache {
namespace server {

// ============================================================================
// Monotonic clock helper
// ============================================================================

static uint64_t now_monotonic_us() {
  using Clock = std::chrono::steady_clock;
  auto now = Clock::now();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          now.time_since_epoch())
          .count());
}

// ============================================================================
// Slot — per-object state
// ============================================================================

struct Slot {
  enum class State { FREE, WRITING, READY, READING };

  State state = State::FREE;
  void* data = nullptr;
  size_t size_bytes = 0;
  MemoryLayoutDesc layout;
  lmcache::storage_manager::TTLLock read_lock;
  uint64_t last_access_time = 0;

  // Index into the slab free list (the offset within the mmap region).
  size_t slab_offset = 0;

  explicit Slot(uint32_t ttl_seconds = 300) : read_lock(ttl_seconds) {}
};

// ============================================================================
// FreeBlock — bump allocator + free list for slab memory
// ============================================================================

struct FreeBlock {
  size_t offset;
  size_t size;
};

class SlabAllocator {
 public:
  SlabAllocator(size_t capacity) : capacity_(capacity), bump_(0) {}

  // Allocate `size` bytes. Returns offset or (size_t)-1 on failure.
  size_t allocate(size_t size) {
    if (size == 0) return 0;

    // Round up to 64-byte alignment for cache-line / DMA friendliness
    size = align_up(size, 64);

    // First-fit from free list
    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
      if (it->size >= size) {
        size_t offset = it->offset;
        if (it->size == size) {
          free_list_.erase(it);
        } else {
          it->offset += size;
          it->size -= size;
        }
        used_ += size;
        return offset;
      }
    }

    // Bump allocate
    if (bump_ + size <= capacity_) {
      size_t offset = bump_;
      bump_ += size;
      used_ += size;
      return offset;
    }

    return static_cast<size_t>(-1);  // OOM
  }

  void deallocate(size_t offset, size_t size) {
    if (size == 0) return;
    size = align_up(size, 64);
    free_list_.push_back({offset, size});
    used_ -= size;
    coalesce_last();
  }

  size_t used() const { return used_; }
  size_t capacity() const { return capacity_; }

 private:
  static size_t align_up(size_t v, size_t a) { return (v + a - 1) & ~(a - 1); }

  // Try to merge the last inserted free block with neighbours.
  // Simple O(n) sweep — good enough for typical workloads.
  void coalesce_last() {
    if (free_list_.size() < 2) return;

    // Sort by offset, then merge adjacent
    std::sort(free_list_.begin(), free_list_.end(),
              [](const FreeBlock& a, const FreeBlock& b) {
                return a.offset < b.offset;
              });

    std::deque<FreeBlock> merged;
    merged.push_back(free_list_.front());
    for (size_t i = 1; i < free_list_.size(); ++i) {
      auto& back = merged.back();
      if (back.offset + back.size == free_list_[i].offset) {
        back.size += free_list_[i].size;
      } else {
        merged.push_back(free_list_[i]);
      }
    }

    // Trim bump pointer if the last free block is contiguous with it
    if (!merged.empty()) {
      auto& last = merged.back();
      if (last.offset + last.size == bump_) {
        bump_ = last.offset;
        merged.pop_back();
      }
    }

    free_list_ = std::move(merged);
  }

  size_t capacity_;
  size_t bump_ = 0;
  size_t used_ = 0;
  std::deque<FreeBlock> free_list_;
};

// ============================================================================
// L1StoreImpl
// ============================================================================

class L1StoreImpl : public L1Store {
 public:
  explicit L1StoreImpl(const L1StoreConfig& config)
      : config_(config), allocator_(config.capacity_bytes) {
    // mmap the slab region
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (config.use_hugepages) {
      flags |= MAP_HUGETLB;
    }

    slab_ = ::mmap(nullptr, config.capacity_bytes, PROT_READ | PROT_WRITE,
                   flags, -1, 0);
    if (slab_ == MAP_FAILED) {
      // Retry without hugepages
      if (config.use_hugepages) {
        flags &= ~MAP_HUGETLB;
        slab_ = ::mmap(nullptr, config.capacity_bytes, PROT_READ | PROT_WRITE,
                       flags, -1, 0);
      }
      if (slab_ == MAP_FAILED) {
        throw std::runtime_error("L1StoreImpl: mmap failed for " +
                                 std::to_string(config.capacity_bytes) +
                                 " bytes");
      }
    }

    // Optionally pin for zero-copy GPU DMA
    if (config.cuda_host_register) {
      cudaError_t err = cudaHostRegister(slab_, config.capacity_bytes,
                                         cudaHostRegisterDefault);
      if (err != cudaSuccess) {
        // Non-fatal: log and continue without pinning
        cuda_registered_ = false;
      } else {
        cuda_registered_ = true;
      }
    }
  }

  ~L1StoreImpl() override {
    // Force-clear to release all state
    clear(/*force=*/true);

    if (cuda_registered_) {
      cudaHostUnregister(slab_);
    }
    ::munmap(slab_, config_.capacity_bytes);
  }

  // ------------------------------------------------------------------
  // Write path
  // ------------------------------------------------------------------

  std::unordered_map<ObjectKey, MemorySlabRef, ObjectKeyHash> reserve_write(
      const std::vector<ObjectKey>& keys, const MemoryLayoutDesc& layout,
      const std::string& mode) override {
    (void)mode;

    // Compute per-object size from layout
    size_t obj_size = compute_layout_size(layout);

    std::lock_guard<std::mutex> lk(mu_);

    std::unordered_map<ObjectKey, MemorySlabRef, ObjectKeyHash> result;

    for (const auto& key : keys) {
      // If key already exists and is not FREE, skip
      auto it = key_to_slot_.find(key);
      if (it != key_to_slot_.end()) {
        auto& slot = slots_[it->second];
        if (slot.state != Slot::State::FREE) {
          continue;  // already in use
        }
        // Recycle: free the old slab region if sizes differ
        if (slot.size_bytes != obj_size) {
          allocator_.deallocate(slot.slab_offset, slot.size_bytes);
          size_t offset = allocator_.allocate(obj_size);
          if (offset == static_cast<size_t>(-1)) {
            // Try eviction
            evict_locked(obj_size);
            offset = allocator_.allocate(obj_size);
            if (offset == static_cast<size_t>(-1)) continue;  // still OOM
          }
          slot.slab_offset = offset;
          slot.data = static_cast<char*>(slab_) + offset;
          slot.size_bytes = obj_size;
        }
        slot.state = Slot::State::WRITING;
        slot.layout = layout;
        slot.last_access_time = now_monotonic_us();
        result[key] = MemorySlabRef{slot.data, slot.size_bytes, slot.layout};
        continue;
      }

      // Need to allocate a new slot
      size_t offset = allocator_.allocate(obj_size);
      if (offset == static_cast<size_t>(-1)) {
        evict_locked(obj_size);
        offset = allocator_.allocate(obj_size);
        if (offset == static_cast<size_t>(-1)) continue;  // OOM
      }

      size_t slot_idx = alloc_slot_index();
      auto& slot = slots_[slot_idx];
      slot.state = Slot::State::WRITING;
      slot.slab_offset = offset;
      slot.data = static_cast<char*>(slab_) + offset;
      slot.size_bytes = obj_size;
      slot.layout = layout;
      slot.last_access_time = now_monotonic_us();

      key_to_slot_[key] = slot_idx;
      result[key] = MemorySlabRef{slot.data, slot.size_bytes, slot.layout};
    }

    return result;
  }

  void finish_write(const std::vector<ObjectKey>& keys) override {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& key : keys) {
      auto it = key_to_slot_.find(key);
      if (it == key_to_slot_.end()) continue;
      auto& slot = slots_[it->second];
      if (slot.state == Slot::State::WRITING) {
        slot.state = Slot::State::READY;
        slot.last_access_time = now_monotonic_us();
      }
    }
  }

  // ------------------------------------------------------------------
  // Read path
  // ------------------------------------------------------------------

  std::unordered_map<ObjectKey, MemorySlabRef, ObjectKeyHash> reserve_read(
      const std::vector<ObjectKey>& keys, int extra_count) override {
    std::lock_guard<std::mutex> lk(mu_);

    std::unordered_map<ObjectKey, MemorySlabRef, ObjectKeyHash> result;

    for (const auto& key : keys) {
      auto it = key_to_slot_.find(key);
      if (it == key_to_slot_.end()) continue;

      auto& slot = slots_[it->second];
      if (slot.state != Slot::State::READY &&
          slot.state != Slot::State::READING) {
        continue;
      }

      // Lock for this reader + extra_count (MLA multi-reader)
      int lock_count = 1 + extra_count;
      for (int i = 0; i < lock_count; ++i) {
        slot.read_lock.lock();
      }
      slot.state = Slot::State::READING;
      slot.last_access_time = now_monotonic_us();

      result[key] = MemorySlabRef{slot.data, slot.size_bytes, slot.layout};
    }

    return result;
  }

  void finish_read(const std::vector<ObjectKey>& keys,
                   int extra_count) override {
    std::lock_guard<std::mutex> lk(mu_);

    for (const auto& key : keys) {
      auto it = key_to_slot_.find(key);
      if (it == key_to_slot_.end()) continue;

      auto& slot = slots_[it->second];
      if (slot.state != Slot::State::READING) continue;

      int unlock_count = 1 + extra_count;
      for (int i = 0; i < unlock_count; ++i) {
        slot.read_lock.unlock();
      }

      // Transition back to READY if no longer locked
      if (!slot.read_lock.is_locked()) {
        slot.state = Slot::State::READY;
      }
    }
  }

  // ------------------------------------------------------------------
  // Lookup
  // ------------------------------------------------------------------

  int64_t prefix_lookup(const std::vector<ObjectKey>& keys) override {
    std::lock_guard<std::mutex> lk(mu_);

    int64_t count = 0;
    for (const auto& key : keys) {
      auto it = key_to_slot_.find(key);
      if (it == key_to_slot_.end()) break;
      auto& slot = slots_[it->second];
      if (slot.state != Slot::State::READY &&
          slot.state != Slot::State::READING) {
        break;
      }
      ++count;
    }
    return count;
  }

  // ------------------------------------------------------------------
  // Deletion / eviction
  // ------------------------------------------------------------------

  L1Error delete_key(const ObjectKey& key) override {
    std::lock_guard<std::mutex> lk(mu_);
    return delete_key_locked(key);
  }

  size_t evict(size_t bytes_needed) override {
    std::lock_guard<std::mutex> lk(mu_);
    return evict_locked(bytes_needed);
  }

  // ------------------------------------------------------------------
  // Capacity / diagnostics
  // ------------------------------------------------------------------

  size_t total_capacity_bytes() const override {
    return config_.capacity_bytes;
  }

  size_t used_bytes() const override {
    std::lock_guard<std::mutex> lk(mu_);
    return allocator_.used();
  }

  void clear(bool force) override {
    std::lock_guard<std::mutex> lk(mu_);

    // Collect all keys to delete
    std::vector<ObjectKey> to_delete;
    to_delete.reserve(key_to_slot_.size());
    for (const auto& [key, idx] : key_to_slot_) {
      auto& slot = slots_[idx];
      if (!force && slot.read_lock.is_locked()) continue;
      to_delete.push_back(key);
    }
    for (const auto& key : to_delete) {
      delete_key_locked_force(key, force);
    }
  }

  bool memcheck() const override {
    std::lock_guard<std::mutex> lk(mu_);

    // Check 1: every entry in key_to_slot_ points to a valid slot
    for (const auto& [key, idx] : key_to_slot_) {
      if (idx >= slots_.size()) return false;
      auto& slot = slots_[idx];
      if (slot.state == Slot::State::FREE)
        return false;  // mapped key shouldn't be FREE
      if (slot.data == nullptr) return false;
      // data pointer should be within slab region
      auto* p = static_cast<char*>(slot.data);
      auto* base = static_cast<char*>(slab_);
      if (p < base || p >= base + config_.capacity_bytes) return false;
      if (slot.slab_offset + slot.size_bytes > config_.capacity_bytes)
        return false;
    }

    // Check 2: free slot indices should have FREE state
    for (size_t fi : free_slot_indices_) {
      if (fi >= slots_.size()) return false;
      if (slots_[fi].state != Slot::State::FREE) return false;
    }

    return true;
  }

 private:
  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  size_t alloc_slot_index() {
    if (!free_slot_indices_.empty()) {
      size_t idx = free_slot_indices_.back();
      free_slot_indices_.pop_back();
      return idx;
    }
    slots_.emplace_back(config_.ttl_seconds);
    return slots_.size() - 1;
  }

  void free_slot_index(size_t idx) {
    auto& slot = slots_[idx];
    slot.state = Slot::State::FREE;
    slot.data = nullptr;
    slot.size_bytes = 0;
    slot.layout = {};
    slot.read_lock.reset();
    slot.last_access_time = 0;
    slot.slab_offset = 0;
    free_slot_indices_.push_back(idx);
  }

  L1Error delete_key_locked(const ObjectKey& key) {
    return delete_key_locked_force(key, /*force=*/false);
  }

  L1Error delete_key_locked_force(const ObjectKey& key, bool force) {
    auto it = key_to_slot_.find(key);
    if (it == key_to_slot_.end()) return L1Error::KEY_NOT_EXIST;

    size_t idx = it->second;
    auto& slot = slots_[idx];

    if (!force && slot.read_lock.is_locked()) {
      return L1Error::KEY_IS_LOCKED;
    }

    if (!force && slot.state == Slot::State::WRITING) {
      return L1Error::KEY_IN_WRONG_STATE;
    }

    // Release slab memory
    allocator_.deallocate(slot.slab_offset, slot.size_bytes);

    // Return slot to free list
    free_slot_index(idx);
    key_to_slot_.erase(it);

    return L1Error::SUCCESS;
  }

  // LRU eviction: evict READY slots with oldest access time.
  // Returns total bytes freed.
  size_t evict_locked(size_t bytes_needed) {
    size_t freed = 0;

    while (freed < bytes_needed) {
      // Find the READY slot with oldest last_access_time
      ObjectKey victim_key;
      size_t victim_idx = static_cast<size_t>(-1);
      uint64_t oldest_time = UINT64_MAX;

      for (const auto& [key, idx] : key_to_slot_) {
        auto& slot = slots_[idx];
        if (slot.state == Slot::State::READY && !slot.read_lock.is_locked() &&
            slot.last_access_time < oldest_time) {
          oldest_time = slot.last_access_time;
          victim_idx = idx;
          victim_key = key;
        }
      }

      if (victim_idx == static_cast<size_t>(-1)) {
        break;  // No more evictable slots
      }

      size_t victim_size = slots_[victim_idx].size_bytes;
      delete_key_locked_force(victim_key, /*force=*/false);
      freed += victim_size;
    }

    return freed;
  }

  /// Compute total byte size from a MemoryLayoutDesc
  static size_t compute_layout_size(const MemoryLayoutDesc& layout) {
    size_t total = 0;
    for (size_t i = 0; i < layout.shapes.size(); ++i) {
      size_t elem_count = 1;
      for (auto d : layout.shapes[i].dims) {
        elem_count *= static_cast<size_t>(d);
      }
      DType dt = (i < layout.dtypes.size()) ? layout.dtypes[i] : DType::Float16;
      total += elem_count * dtype_size(dt);
    }
    return total;
  }

  // ------------------------------------------------------------------
  // Data members
  // ------------------------------------------------------------------

  L1StoreConfig config_;
  void* slab_ = nullptr;
  bool cuda_registered_ = false;

  mutable std::mutex mu_;
  SlabAllocator allocator_;

  // deque: Slot contains std::atomic (TTLLock) so it's non-movable.
  // deque never relocates existing elements on push_back.
  std::deque<Slot> slots_;
  std::vector<size_t> free_slot_indices_;
  std::unordered_map<ObjectKey, size_t, ObjectKeyHash> key_to_slot_;
};

// ============================================================================
// Factory
// ============================================================================

std::unique_ptr<L1Store> L1Store::create(const L1StoreConfig& config) {
  return std::make_unique<L1StoreImpl>(config);
}

}  // namespace server
}  // namespace lmcache
