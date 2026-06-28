// SPDX-License-Identifier: Apache-2.0

#include "ttl_lock.h"

#include <limits>
#include <stdexcept>
#include <thread>

namespace lmcache {
namespace storage_manager {

TTLLock::TTLLock(uint32_t ttl_sec)
    : counter_(0),
      expiration_ms_(0),
      ttl_ms_(static_cast<int64_t>(ttl_sec) * 1000) {}

void TTLLock::lock() { lock_count(1); }

void TTLLock::lock_count(int64_t count) {
  if (count <= 0) {
    throw std::invalid_argument("TTLLock::lock_count count must be positive");
  }

  int64_t current_time = now_ms();
  int64_t new_expiration = current_time + ttl_ms_;

  // Use compare-and-swap loop to handle the TTL expiration case
  while (true) {
    int64_t old_counter = counter_.load(std::memory_order_acquire);
    int64_t old_expiration = expiration_ms_.load(std::memory_order_acquire);

    if (old_counter < 0) {
      // Another thread is resetting an expired counter. Retry until the
      // reset publishes a non-negative count.
      std::this_thread::yield();
      continue;
    }

    // Check if TTL has expired
    bool expired = (current_time >= old_expiration);

    if (expired) {
      // TTL expired: claim reset ownership by moving the non-negative counter
      // to a sentinel value. This prevents concurrent lock_count() calls from
      // incrementing a stale counter that would then be overwritten.
      if (counter_.compare_exchange_strong(old_counter, -1,
                                           std::memory_order_seq_cst)) {
        expiration_ms_.store(new_expiration, std::memory_order_seq_cst);
        counter_.store(count, std::memory_order_seq_cst);
        return;
      }
      // Another thread modified the counter, retry
      continue;
    } else {
      // TTL not expired, try to increment counter
      if (old_counter > std::numeric_limits<int64_t>::max() - count) {
        throw std::overflow_error("TTLLock::lock_count counter overflow");
      }
      if (counter_.compare_exchange_strong(old_counter, old_counter + count,
                                           std::memory_order_seq_cst)) {
        // Successfully incremented counter, update expiration
        expiration_ms_.store(new_expiration, std::memory_order_seq_cst);
        return;
      }
      // Another thread modified counter, retry
      continue;
    }
  }
}

void TTLLock::unlock() { unlock_count(1); }

void TTLLock::unlock_count(int64_t count) {
  if (count <= 0) {
    throw std::invalid_argument("TTLLock::unlock_count count must be positive");
  }

  // Use compare-and-swap loop to ensure we don't go below 0
  while (true) {
    int64_t old_counter = counter_.load(std::memory_order_acquire);

    if (old_counter < 0) {
      // A lock_count() call is resetting an expired counter.
      std::this_thread::yield();
      continue;
    }

    if (old_counter <= 0) {
      // Already at 0, nothing to do
      return;
    }

    int64_t new_counter = old_counter > count ? old_counter - count : 0;
    if (counter_.compare_exchange_strong(old_counter, new_counter,
                                         std::memory_order_seq_cst)) {
      return;
    }
    // Another thread modified counter, retry
  }
}

bool TTLLock::is_locked() const {
  int64_t current_time = now_ms();
  int64_t expiration = expiration_ms_.load(std::memory_order_acquire);
  int64_t counter = counter_.load(std::memory_order_acquire);

  // Lock is held if counter > 0 AND TTL not expired
  return (counter > 0) && (current_time < expiration);
}

void TTLLock::reset() {
  counter_.store(0, std::memory_order_seq_cst);
  expiration_ms_.store(0, std::memory_order_seq_cst);
}

int64_t TTLLock::now_ms() {
  auto now = Clock::now();
  return to_ms(now);
}

int64_t TTLLock::to_ms(const TimePoint& tp) {
  return static_cast<int64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          tp.time_since_epoch())
          .count());
}

}  // namespace storage_manager
}  // namespace lmcache
