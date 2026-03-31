// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — session management implementation

#include "session_manager.h"

#include <cassert>
#include <chrono>

namespace lmcache {
namespace server {

// ----------------------------------------------------------------------------
// Helper: current time in seconds (steady_clock)
// ----------------------------------------------------------------------------

namespace {

double steady_now() {
  auto tp = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(tp.time_since_epoch()).count();
}

}  // namespace

// ----------------------------------------------------------------------------
// Session
// ----------------------------------------------------------------------------

Session::Session(const std::string& request_id, const TokenHasher& hasher)
    : request_id_(request_id),
      hasher_(hasher),
      created_at_(steady_now()),
      last_prefix_hash_(hasher.none_hash()),
      num_chunks_processed_(0) {}

void Session::set_tokens(const std::vector<int32_t>& full_token_ids) {
  std::lock_guard<std::mutex> lock(mutex_);
  token_ids_ = full_token_ids;
}

std::vector<HashBytes> Session::get_hashes(int start, int end) {
  int chunk_size = hasher_.chunk_size();
  assert(start % chunk_size == 0 && "start must be a multiple of chunk_size");
  assert(end % chunk_size == 0 && "end must be a multiple of chunk_size");

  int start_chunk = start / chunk_size;
  int end_chunk = end / chunk_size;

  std::lock_guard<std::mutex> lock(mutex_);
  compute_hash_up_to(end_chunk);

  // Return the slice [start_chunk, end_chunk)
  std::vector<HashBytes> result;
  if (start_chunk < end_chunk &&
      start_chunk < static_cast<int>(chunk_hashes_.size())) {
    int actual_end =
        std::min(end_chunk, static_cast<int>(chunk_hashes_.size()));
    result.assign(chunk_hashes_.begin() + start_chunk,
                  chunk_hashes_.begin() + actual_end);
  }
  return result;
}

void Session::compute_hash_up_to(int end_chunk) {
  // Caller must hold mutex_
  int chunk_size = hasher_.chunk_size();

  while (num_chunks_processed_ < end_chunk) {
    int cs = num_chunks_processed_ * chunk_size;
    int ce = cs + chunk_size;

    // Ensure we have enough tokens
    if (ce > static_cast<int>(token_ids_.size())) {
      break;
    }

    HashBytes h =
        hasher_.hash_tokens(token_ids_.data() + cs,
                            static_cast<size_t>(chunk_size), last_prefix_hash_);

    last_prefix_hash_ = h;
    chunk_hashes_.push_back(std::move(h));
    num_chunks_processed_++;
  }
}

// ----------------------------------------------------------------------------
// SessionManager
// ----------------------------------------------------------------------------

SessionManager::SessionManager(const TokenHasher& hasher, double ttl)
    : hasher_(hasher), ttl_(ttl) {}

std::shared_ptr<Session> SessionManager::get_or_create(
    const std::string& request_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = sessions_.find(request_id);
  if (it != sessions_.end()) {
    return it->second;
  }
  auto session = std::make_shared<Session>(request_id, hasher_);
  sessions_.emplace(request_id, session);
  return session;
}

void SessionManager::remove(const std::string& request_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sessions_.erase(request_id);
}

int SessionManager::cleanup_expired() {
  double now = steady_now();
  std::lock_guard<std::mutex> lock(mutex_);

  int removed = 0;
  for (auto it = sessions_.begin(); it != sessions_.end();) {
    if (now - it->second->created_at() > ttl_) {
      it = sessions_.erase(it);
      ++removed;
    } else {
      ++it;
    }
  }
  return removed;
}

int SessionManager::active_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(sessions_.size());
}

}  // namespace server
}  // namespace lmcache
