// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace lmcache {
namespace storage_manager {

/**
 * @brief A thread-safe lock with TTL (Time-To-Live) support.
 *
 * This lock maintains a counter that can be incremented (lock) and decremented
 * (unlock). The lock also has a TTL mechanism - if the TTL expires, the lock
 * is considered unlocked regardless of the counter value.
 *
 * Thread Safety:
 * - All operations are thread-safe using atomic operations.
 * - lock() and unlock() use memory_order_seq_cst for full synchronization.
 * - is_locked() uses memory_order_acquire for visibility guarantees.
 */
class TTLLock {
 public:
  /**
   * @brief Construct a new TTLLock with the specified TTL duration.
   *
   * @param ttl_second The TTL duration in seconds. Default is 300 seconds.
   */
  explicit TTLLock(uint32_t ttl_second = 300);

  /**
   * @brief Increment the lock counter by 1 and update the TTL.
   *
   * If the previous TTL has expired, the counter is reset to 1.
   * Otherwise, the counter is incremented by 1.
   * The TTL is always refreshed to the current time + TTL duration.
   */
  void lock();

  /**
   * @brief Increment the lock counter by count and update the TTL.
   *
   * If the previous TTL has expired, the counter is reset to count.
   * Otherwise, the counter is incremented by count.
   * The TTL is always refreshed to the current time + TTL duration.
   *
   * @param count Number of lock counts to acquire. Must be positive.
   *
   * @throws std::invalid_argument if count is not positive.
   * @throws std::overflow_error if the counter would overflow.
   */
  void lock_count(int64_t count);

  /**
   * @brief Decrement the lock counter by 1.
   *
   * The counter will not go below 0.
   */
  void unlock();

  /**
   * @brief Decrement the lock counter by count.
   *
   * The counter will not go below 0.
   *
   * @param count Number of lock counts to release. Must be positive.
   *
   * @throws std::invalid_argument if count is not positive.
   */
  void unlock_count(int64_t count);

  /**
   * @brief Check if the lock is currently held.
   *
   * The lock is considered held if:
   * 1. The counter is greater than 0, AND
   * 2. The TTL has not expired.
   *
   * @return true if the lock is held, false otherwise.
   */
  bool is_locked() const;

  /**
   * @brief Reset the lock to initial state (counter = 0, TTL expired).
   */
  void reset();

 private:
  using Clock = std::chrono::steady_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  /**
   * @brief Get the current timestamp in milliseconds since epoch.
   */
  static int64_t now_ms();

  /**
   * @brief Convert TimePoint to milliseconds since epoch.
   */
  static int64_t to_ms(const TimePoint& tp);

  // The lock counter
  std::atomic<int64_t> counter_;

  // The expiration time in milliseconds since steady_clock epoch
  std::atomic<int64_t> expiration_ms_;

  // The TTL duration in milliseconds
  const int64_t ttl_ms_;
};

}  // namespace storage_manager
}  // namespace lmcache
