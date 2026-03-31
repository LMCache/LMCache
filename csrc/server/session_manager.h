// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — Per-request session tracking
//
// Caches computed chunk hashes per request_id so that repeated
// store/retrieve/lookup operations avoid redundant hashing.
// Thread-safe: multiple TP workers may access the same session.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "token_hasher.h"

namespace lmcache {
namespace server {

// ============================================================================
// Session — per-request hash cache
// ============================================================================

class Session {
 public:
  Session(const std::string& request_id, const TokenHasher& hasher);
  ~Session() = default;

  /// Update the full token sequence (idempotent, replaces not extends).
  void set_tokens(const std::vector<int32_t>& full_token_ids);

  /// Compute and return chunk hashes for [start, end) token range.
  /// Internally caches rolling hashes up to end, skipping already-computed.
  /// start and end must be chunk-aligned.
  std::vector<HashBytes> get_hashes(int start, int end);

  const std::string& request_id() const { return request_id_; }

  /// Creation timestamp (steady clock, seconds).
  double created_at() const { return created_at_; }

 private:
  void compute_hash_up_to(int end_chunk);

  std::string request_id_;
  const TokenHasher& hasher_;
  double created_at_;

  // Protected by mutex_
  mutable std::mutex mutex_;
  std::vector<int32_t> token_ids_;
  std::vector<HashBytes> chunk_hashes_;
  HashBytes last_prefix_hash_;
  int num_chunks_processed_ = 0;
};

// ============================================================================
// SessionManager — thread-safe manager for per-request sessions
// ============================================================================

class SessionManager {
 public:
  static constexpr double DEFAULT_SESSION_TTL = 600.0;  // 10 minutes

  /// @param hasher  Reference to the shared TokenHasher (must outlive this)
  /// @param ttl     Session TTL in seconds
  SessionManager(const TokenHasher& hasher, double ttl = DEFAULT_SESSION_TTL);
  ~SessionManager() = default;

  /// Get existing session or create a new one.
  std::shared_ptr<Session> get_or_create(const std::string& request_id);

  /// Remove a session by request_id.
  void remove(const std::string& request_id);

  /// Remove sessions that have exceeded their TTL.
  /// Returns number of sessions removed.
  int cleanup_expired();

  /// Number of currently tracked sessions.
  int active_count() const;

 private:
  const TokenHasher& hasher_;
  double ttl_;

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
};

}  // namespace server
}  // namespace lmcache
