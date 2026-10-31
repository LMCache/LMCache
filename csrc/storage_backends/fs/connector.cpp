// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include "../keys.h"

namespace lmcache {
namespace connector {

// ---------------------------------------------------------------
// read/write helpers
// ---------------------------------------------------------------

static void write_all(int fd, const void* data, size_t len) {
  size_t written = 0;
  const char* ptr = static_cast<const char*>(data);
  while (written < len) {
    ssize_t n = ::write(fd, ptr + written, len - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("write failed: " + std::string(strerror(errno)));
    }
    if (n == 0) {
      throw std::runtime_error("write returned 0");
    }
    written += static_cast<size_t>(n);
  }
}

static size_t read_all(int fd, void* buf, size_t len) {
  size_t total = 0;
  char* ptr = static_cast<char*>(buf);
  while (total < len) {
    ssize_t n = ::read(fd, ptr + total, len - total);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

// pread loop: descriptors shared through HeldFdCache have no usable file
// offset, so every read names its own.
static size_t pread_all(int fd, void* buf, size_t len, off_t offset) {
  size_t total = 0;
  char* ptr = static_cast<char*>(buf);
  while (total < len) {
    ssize_t n = ::pread(fd, ptr + total, len - total,
                        offset + static_cast<off_t>(total));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("read failed: " + std::string(strerror(errno)));
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

static bool try_enable_odirect(int& flags, const void* buf, size_t len,
                               size_t disk_block_size) {
#ifdef O_DIRECT
  if (disk_block_size == 0 || len % disk_block_size != 0) {
    return false;
  }
  auto addr = reinterpret_cast<std::uintptr_t>(buf);
  if (addr % disk_block_size != 0) {
    throw std::runtime_error(
        "O_DIRECT buffer address is not aligned to filesystem block size");
  }
  flags |= O_DIRECT;
  return true;
#else
  (void)flags;
  (void)buf;
  (void)len;
  (void)disk_block_size;
  return false;
#endif
}

// ---------------------------------------------------------------
// HeldFdCache
// ---------------------------------------------------------------

std::string HeldFdCache::key_of(const std::string& name, int flags) {
#ifdef O_DIRECT
  const bool direct = (flags & O_DIRECT) != 0;
#else
  const bool direct = false;
  (void)flags;
#endif
  return name + (direct ? "|d" : "|b");
}

void HeldFdCache::retire_locked(Entry& e) {
  if (e.refs == 0) {
    ::close(e.fd);
  } else {
    retired_[e.fd] = e.refs;
  }
}

void HeldFdCache::evict_locked() {
  // Least recently used first; entries in use are skipped, so the cache
  // can exceed its capacity only by the descriptors reads are using.
  auto it = lru_.end();
  while (entries_.size() > capacity_ && it != lru_.begin()) {
    --it;
    auto e = entries_.find(*it);
    if (e->second.refs != 0) {
      continue;
    }
    ::close(e->second.fd);
    entries_.erase(e);
    it = lru_.erase(it);
    ++evictions_;
  }
}

int HeldFdCache::acquire(const std::string& name,
                         const std::filesystem::path& path, int flags,
                         size_t min_size) {
  const std::string key = key_of(name, flags);
  {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = entries_.find(key);
    if (it != entries_.end()) {
      ++it->second.refs;
      lru_.splice(lru_.begin(), lru_, it->second.lru);
      ++hits_;
      return it->second.fd;
    }
  }
  int fd = ::open(path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + path.string() + ": " +
                             strerror(errno));
  }
  struct stat st{};
  if (::fstat(fd, &st) != 0 || static_cast<size_t>(st.st_size) < min_size) {
    const int err = errno;
    const auto size = static_cast<long long>(st.st_size);
    ::close(fd);
    throw std::runtime_error("object too short for " + path.string() + ": " +
                             std::to_string(size) + " bytes" +
                             (err ? std::string(": ") + strerror(err) : ""));
  }
  std::lock_guard<std::mutex> lk(mu_);
  ++opens_;
  auto it = entries_.find(key);
  if (it != entries_.end()) {
    // Another reader opened it meanwhile; keep one descriptor.
    ::close(fd);
    ++races_;
    ++it->second.refs;
    lru_.splice(lru_.begin(), lru_, it->second.lru);
    return it->second.fd;
  }
  lru_.push_front(key);
  Entry e;
  e.fd = fd;
  e.refs = 1;
  e.lru = lru_.begin();
  entries_.emplace(key, e);
  evict_locked();
  return fd;
}

void HeldFdCache::release(const std::string& name, int flags, int fd,
                          bool stale) {
  std::lock_guard<std::mutex> lk(mu_);
  auto r = retired_.find(fd);
  if (r != retired_.end()) {
    if (--r->second == 0) {
      ::close(fd);
      retired_.erase(r);
      ++deferred_closes_;
    }
    return;
  }
  auto it = entries_.find(key_of(name, flags));
  if (it == entries_.end() || it->second.fd != fd) {
    return;  // cannot happen: a referenced entry is cached or retired
  }
  --it->second.refs;
  if (stale) {
    ++stale_drops_;
    lru_.erase(it->second.lru);
    Entry e = it->second;
    entries_.erase(it);
    retire_locked(e);
    return;
  }
  evict_locked();
}

bool HeldFdCache::contains(const std::string& name) {
  std::lock_guard<std::mutex> lk(mu_);
  return entries_.count(name + "|b") != 0 || entries_.count(name + "|d") != 0;
}

void HeldFdCache::invalidate(const std::string& name) {
  std::lock_guard<std::mutex> lk(mu_);
  for (const char* suffix : {"|b", "|d"}) {
    auto it = entries_.find(name + suffix);
    if (it == entries_.end()) {
      continue;
    }
    ++invalidations_;
    lru_.erase(it->second.lru);
    Entry e = it->second;
    entries_.erase(it);
    retire_locked(e);
  }
}

void HeldFdCache::close_all() {
  std::lock_guard<std::mutex> lk(mu_);
  for (auto& kv : entries_) {
    ::close(kv.second.fd);
  }
  for (auto& kv : retired_) {
    ::close(kv.first);
  }
  entries_.clear();
  retired_.clear();
  lru_.clear();
}

std::map<std::string, uint64_t> HeldFdCache::stats() {
  std::lock_guard<std::mutex> lk(mu_);
  return {{"opens", opens_},
          {"hits", hits_},
          {"evictions", evictions_},
          {"invalidations", invalidations_},
          {"stale_drops", stale_drops_},
          {"deferred_closes", deferred_closes_},
          {"races", races_},
          {"open_now", entries_.size() + retired_.size()},
          {"capacity", capacity_}};
}

// ---------------------------------------------------------------
// FSConnector
// ---------------------------------------------------------------

// No pool, no budget.  A pool with no figure gets the default.
static size_t effective_read_budget(int num_read_workers, size_t budget) {
  if (num_read_workers <= 0) {
    return 0;
  }
  return budget == 0 ? kDefaultReadMaxBytesInFlight : budget;
}

FSConnector::FSConnector(std::string base_path, int num_workers,
                         std::string relative_tmp_dir, bool use_odirect,
                         size_t read_ahead_size, int num_read_workers,
                         size_t read_max_bytes_in_flight,
                         size_t held_fd_capacity)
    : ConnectorBase(num_workers),
      held_fds_(held_fd_capacity),
      base_path_(std::move(base_path)),
      relative_tmp_dir_(std::move(relative_tmp_dir)),
      use_odirect_(use_odirect),
      disk_block_size_(0),
      read_ahead_size_(read_ahead_size),
      read_max_bytes_in_flight_(
          effective_read_budget(num_read_workers, read_max_bytes_in_flight)) {
  if (num_read_workers < 0) {
    throw std::runtime_error("num_read_workers must be >= 0");
  }

  // Create base directory
  std::filesystem::create_directories(base_path_);

  // Create tmp directory if configured
  if (!relative_tmp_dir_.empty()) {
    auto tmp_path = std::filesystem::path(base_path_) / relative_tmp_dir_;
    std::filesystem::create_directories(tmp_path);
  }

  // Query disk block size for O_DIRECT
  if (use_odirect_) {
    struct statvfs st;
    if (statvfs(base_path_.c_str(), &st) == 0) {
      disk_block_size_ = st.f_bsize;
    }
  }

  // Readers first.  On failure join whatever started: the destructor does
  // not run for a half-built object.
  try {
    start_read_pool(num_read_workers);
    start_workers();  // IMPORTANT: call at END of constructor
  } catch (...) {
    close();
    throw;
  }
}

FSConnector::~FSConnector() { close(); }

WorkerFSConn FSConnector::create_connection() {
  WorkerFSConn conn;
  conn.base_path = base_path_;
  if (!relative_tmp_dir_.empty()) {
    conn.tmp_dir = std::filesystem::path(base_path_) / relative_tmp_dir_;
  }
  conn.use_odirect = use_odirect_;
  conn.disk_block_size = disk_block_size_;
  conn.read_ahead_size = read_ahead_size_;
  return conn;
}

void FSConnector::do_single_get(WorkerFSConn& conn, const std::string& key,
                                void* buf, size_t len, size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  int flags = O_RDONLY;
  bool do_odirect = conn.use_odirect &&
                    try_enable_odirect(flags, buf, len, conn.disk_block_size);

  if (held_fds_.enabled()) {
    get_with_held_fd(conn, filename, file_path, flags, do_odirect, buf, len);
    return;
  }

  int fd = ::open(file_path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + file_path.string() +
                             ": " + strerror(errno));
  }

  try {
    size_t n;
    bool use_read_ahead =
        !do_odirect && conn.read_ahead_size > 0 && len > conn.read_ahead_size;
    if (use_read_ahead) {
      // Trigger filesystem readahead with a small initial
      // read, then read the remainder.
      size_t ra = conn.read_ahead_size;
      size_t n_head = read_all(fd, buf, ra);
      if (n_head < ra) {
        // Short read on the head portion — treat as
        // incomplete
        n = n_head;
      } else {
        size_t n_tail = read_all(fd, static_cast<char*>(buf) + ra, len - ra);
        n = n_head + n_tail;
      }
    } else {
      n = read_all(fd, buf, len);
    }
    if (n != len) {
      throw std::runtime_error("incomplete read for " + file_path.string() +
                               ": expected " + std::to_string(len) + ", got " +
                               std::to_string(n));
    }
  } catch (...) {
    ::close(fd);
    throw;
  }
  ::close(fd);
}

void FSConnector::get_with_held_fd(const WorkerFSConn& conn,
                                   const std::string& filename,
                                   const std::filesystem::path& file_path,
                                   int flags, bool do_odirect, void* buf,
                                   size_t len) {
  for (int attempt = 0;; ++attempt) {
    // Throws on a missing or short file: a miss, as without the cache.
    int fd = held_fds_.acquire(filename, file_path, flags, len);
    size_t n = 0;
    std::string error;
    try {
      bool use_read_ahead =
          !do_odirect && conn.read_ahead_size > 0 && len > conn.read_ahead_size;
      if (use_read_ahead) {
        size_t ra = conn.read_ahead_size;
        n = pread_all(fd, buf, ra, 0);
        if (n == ra) {
          n += pread_all(fd, static_cast<char*>(buf) + ra, len - ra,
                         static_cast<off_t>(ra));
        }
      } else {
        n = pread_all(fd, buf, len, 0);
      }
      if (n != len) {
        error = "incomplete read for " + file_path.string() + ": expected " +
                std::to_string(len) + ", got " + std::to_string(n);
      }
    } catch (const std::exception& e) {
      error = e.what();
    }
    // A failure through a cached descriptor may be a revoked or replaced
    // file: drop it and try once more with a fresh open.
    held_fds_.release(filename, flags, fd, !error.empty());
    if (error.empty()) {
      return;
    }
    if (attempt > 0) {
      throw std::runtime_error(error);
    }
  }
}

std::map<std::string, uint64_t> FSConnector::held_fd_stats() {
  return held_fds_.stats();
}

void FSConnector::do_single_set(WorkerFSConn& conn, const std::string& key,
                                const void* buf, size_t len,
                                size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  // Skip if already stored on disk
  if (std::filesystem::exists(file_path)) {
    return;
  }

  // Determine temp file path
  std::filesystem::path tmp_path;
  if (!conn.tmp_dir.empty()) {
    tmp_path = conn.tmp_dir / filename;
  } else {
    tmp_path = file_path;
    tmp_path.replace_extension(TMP_EXT);
  }

  int flags = O_CREAT | O_WRONLY | O_TRUNC;
  if (conn.use_odirect) {
    try_enable_odirect(flags, buf, len, conn.disk_block_size);
  }

  int fd = ::open(tmp_path.c_str(), flags, 0644);
  if (fd < 0) {
    throw std::runtime_error("open for write failed: " + tmp_path.string() +
                             ": " + strerror(errno));
  }

  try {
    write_all(fd, buf, len);
  } catch (...) {
    ::close(fd);
    // Clean up temp file on failure
    std::filesystem::remove(tmp_path);
    throw;
  }
  ::close(fd);

  // Atomic rename: tmp -> final
  std::error_code ec;
  std::filesystem::rename(tmp_path, file_path, ec);
  if (ec) {
    // Try to clean up, but prioritize reporting the original error.
    std::error_code remove_ec;
    std::filesystem::remove(tmp_path, remove_ec);
    throw std::runtime_error("rename failed: " + tmp_path.string() + " -> " +
                             file_path.string() + ": " + ec.message());
  }
}

bool FSConnector::do_single_exists(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  // A held descriptor is a published object that can still be read, so
  // the lookup needs no stat (on NFS, no GETATTR round trip).
  if (held_fds_.enabled() && held_fds_.contains(filename)) {
    return true;
  }
  auto file_path = conn.base_path / filename;
  return std::filesystem::exists(file_path);
}

bool FSConnector::do_single_delete(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
  held_fds_.invalidate(filename);
  std::error_code ec;
  return std::filesystem::remove(file_path, ec);
}

// ---------------------------------------------------------------
// Read pool
// ---------------------------------------------------------------

size_t FSConnector::read_budget_bytes() const {
  return read_max_bytes_in_flight_;
}

size_t FSConnector::choose_num_tiles(Op op, size_t num_items) const {
  if (op != Op::BATCH_TILE_GET || read_threads_.empty()) {
    return ConnectorBase::choose_num_tiles(op, num_items);
  }
  return 1;
}

void FSConnector::do_batch_get(WorkerFSConn& conn, const Request& req) {
  if (read_threads_.empty()) {
    ConnectorBase::do_batch_get(conn, req);
    return;
  }
  do_batch_get_pooled(req);
}

void FSConnector::on_workers_stopped() {
  stop_read_pool();
  held_fds_.close_all();
}

bool FSConnector::read_budget_admits(size_t len) const {
  return reads_in_flight_ < kMinReadsInFlight ||
         read_bytes_in_flight_ + len <= read_max_bytes_in_flight_;
}

void FSConnector::do_batch_get_pooled(const Request& req) {
  const size_t num_keys = req.keys.size();
  ReadTile tile;
  tile.ok.assign(num_keys, 1);
  tile.remaining = num_keys;
  // Allocated before anything is published, so nothing below can throw
  // while readers hold pointers into this frame.
  std::vector<ReadTask> tasks(num_keys);

  {
    std::unique_lock<std::mutex> lk(read_mu_);
    for (size_t i = 0; i < num_keys; ++i) {
      const size_t len = req.buf_lens[i];
      if (!read_budget_admits(len)) {
        ++read_budget_waiters_;
        read_budget_cv_.wait(lk,
                             [this, len] { return read_budget_admits(len); });
        --read_budget_waiters_;
      }
      read_bytes_in_flight_ += len;
      ++reads_in_flight_;
      tasks[i] = ReadTask{&req, i, &tile, nullptr};
      if (read_queue_tail_ != nullptr) {
        read_queue_tail_->next = &tasks[i];
      } else {
        read_queue_head_ = &tasks[i];
      }
      read_queue_tail_ = &tasks[i];
      read_cv_.notify_one();
    }
    tile.done_cv.wait(lk, [&tile] { return tile.remaining == 0; });
  }

  for (size_t i = 0; i < num_keys; ++i) {
    req.batch->per_key_results[req.start_idx + i] = tile.ok[i];
  }
}

// Connections are built here so a failure surfaces in the constructor.
void FSConnector::start_read_pool(int num_read_workers) {
  if (num_read_workers <= 0) {
    return;
  }
  const size_t num_threads = static_cast<size_t>(num_read_workers);
  read_conns_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    read_conns_.push_back(create_connection());
  }
  read_threads_.reserve(num_threads);
  for (size_t i = 0; i < num_threads; ++i) {
    read_threads_.emplace_back(
        [this, i] { this->read_thread_main(read_conns_[i]); });
  }
}

// Only after the workers are joined; see on_workers_stopped().
void FSConnector::stop_read_pool() {
  {
    std::lock_guard<std::mutex> lk(read_mu_);
    read_stop_ = true;
  }
  read_cv_.notify_all();
  for (auto& t : read_threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
  read_threads_.clear();
  read_conns_.clear();
}

void FSConnector::read_thread_main(WorkerFSConn& conn) {
  // The lock taken to record a completion is kept to pop the next task, so
  // a busy reader takes read_mu_ once per object, not twice.
  std::unique_lock<std::mutex> lk(read_mu_);
  for (;;) {
    read_cv_.wait(lk,
                  [this] { return read_stop_ || read_queue_head_ != nullptr; });
    if (read_queue_head_ == nullptr) {
      return;  // the predicate admits an empty queue only on stop
    }
    ReadTask* task = read_queue_head_;
    read_queue_head_ = task->next;
    if (read_queue_head_ == nullptr) {
      read_queue_tail_ = nullptr;
    }
    lk.unlock();

    uint8_t ok = 1;
    const Request& req = *task->req;
    const std::string& key = req.keys[task->index];
    try {
      do_single_get(conn, key, req.buf_ptrs[task->index],
                    req.buf_lens[task->index], req.batch_chunk_num_bytes);
    } catch (const std::exception& e) {
      ok = 0;
      fprintf(stderr, "[LMCache GET] key %s failed: %s\n", key.c_str(),
              e.what());
    } catch (...) {
      ok = 0;
      fprintf(stderr, "[LMCache GET] key %s failed: unknown exception\n",
              key.c_str());
    }

    // The submitting worker may destroy `task` and its tile as soon as it
    // takes read_mu_ and sees remaining == 0, so nothing touches them
    // after the notify below; notifying under the lock keeps the tile
    // alive until then.
    lk.lock();
    task->tile->ok[task->index] = ok;
    read_bytes_in_flight_ -= req.buf_lens[task->index];
    --reads_in_flight_;
    if (--task->tile->remaining == 0) {
      task->tile->done_cv.notify_one();
    }
    // Large objects can hold more than the budget, so the free bytes are
    // computed without wrapping.
    const size_t free_bytes =
        read_bytes_in_flight_ < read_max_bytes_in_flight_
            ? read_max_bytes_in_flight_ - read_bytes_in_flight_
            : 0;
    if (read_budget_waiters_ != 0 &&
        (reads_in_flight_ < kMinReadsInFlight ||
         free_bytes >= read_max_bytes_in_flight_ / kReadBudgetWakeSlices)) {
      read_budget_cv_.notify_all();
    }
  }
}

}  // namespace connector
}  // namespace lmcache
