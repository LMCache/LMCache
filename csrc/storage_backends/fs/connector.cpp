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
                         size_t read_max_bytes_in_flight)
    : ConnectorBase(num_workers),
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
    if (statvfs(base_path_.c_str(), &st) != 0) {
      throw std::runtime_error("statvfs failed for " + base_path_ + ": " +
                               std::string(strerror(errno)));
    }
    disk_block_size_ = st.f_bsize;
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
  auto file_path = conn.base_path / filename;
  return std::filesystem::exists(file_path);
}

DeleteResult FSConnector::do_single_delete_result(WorkerFSConn& conn,
                                                  const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
  std::error_code ec;
  if (std::filesystem::remove(file_path, ec)) {
    return DeleteResult::DELETED;
  }
  // remove() returns false without an error only when the file did not exist.
  return ec ? DeleteResult::NOT_DELETED : DeleteResult::NOT_FOUND;
}

// ---------------------------------------------------------------
// Read pool
// ---------------------------------------------------------------

size_t FSConnector::read_budget_bytes() const {
  return read_max_bytes_in_flight_;
}

size_t FSConnector::disk_block_size() const { return disk_block_size_; }

uint64_t FSConnector::submit_batch_get(const std::vector<std::string>& keys,
                                       const std::vector<void*>& bufs,
                                       const std::vector<size_t>& lens,
                                       size_t batch_chunk_num_bytes) {
  if (read_max_bytes_in_flight_ == 0) {
    return ConnectorBase::submit_batch_get(keys, bufs, lens,
                                           batch_chunk_num_bytes);
  }
  validate_batch_inputs(keys, bufs, lens);

  auto batch = new_pool_batch(keys);
  batch->buf_ptrs = bufs;
  batch->buf_lens = lens;
  batch->batch_chunk_num_bytes = batch_chunk_num_bytes;
  const uint64_t future_id = batch->future_id;
  enqueue_pool_batch(get_queue_, std::move(batch));
  return future_id;
}

uint64_t FSConnector::submit_batch_exists(
    const std::vector<std::string>& keys) {
  if (read_max_bytes_in_flight_ == 0) {
    return ConnectorBase::submit_batch_exists(keys);
  }
  if (keys.empty()) {
    throw std::runtime_error("keys list is empty");
  }

  auto batch = new_pool_batch(keys);
  const uint64_t future_id = batch->future_id;
  enqueue_pool_batch(exists_queue_, std::move(batch));
  return future_id;
}

void FSConnector::on_workers_stopped() { stop_read_pool(); }

std::shared_ptr<FSConnector::PoolBatch> FSConnector::new_pool_batch(
    const std::vector<std::string>& keys) {
  auto batch = std::make_shared<PoolBatch>();
  batch->future_id = next_future_id_.fetch_add(1, std::memory_order_relaxed);
  batch->keys = keys;
  batch->ok.assign(keys.size(), 0);
  batch->remaining = keys.size();
  return batch;
}

void FSConnector::enqueue_pool_batch(PoolQueue& queue,
                                     std::shared_ptr<PoolBatch> batch) {
  bool wake = false;
  {
    std::lock_guard<std::mutex> lk(read_mu_);
    if (!read_stop_) {
      queue.push_back(std::move(batch));
      wake = has_ready_read_work();
    }
  }
  if (wake) {
    read_cv_.notify_one();
  }
}

bool FSConnector::read_budget_admits(size_t len) const {
  return reads_in_flight_ < kMinReadsInFlight ||
         read_bytes_in_flight_ + len <= read_max_bytes_in_flight_;
}

bool FSConnector::has_ready_read_work() const {
  if (!exists_queue_.empty()) {
    return true;
  }
  if (get_queue_.empty()) {
    return false;
  }
  const PoolBatch& front = *get_queue_.front();
  return read_budget_admits(front.buf_lens[front.next_idx]);
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

void FSConnector::stop_read_pool() {
  // Runs from on_workers_stopped(), before ConnectorBase closes notifier_.
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
  exists_queue_.clear();
  get_queue_.clear();
  read_threads_.clear();
  read_conns_.clear();
}

void FSConnector::read_thread_main(WorkerFSConn& conn) {
  std::unique_lock<std::mutex> lk(read_mu_);
  for (;;) {
    read_cv_.wait(lk, [this] { return read_stop_ || has_ready_read_work(); });
    if (read_stop_) {
      return;
    }
    if (!exists_queue_.empty()) {
      run_lookup(conn, lk);
    } else {
      run_read(conn, lk);
    }
  }
}

void FSConnector::run_lookup(WorkerFSConn& conn,
                             std::unique_lock<std::mutex>& lk) {
  auto [batch, idx] = claim_next_key(exists_queue_);
  unlock_and_wake_next(lk);

  uint8_t found = 0;
  bool failed = false;
  std::string err;
  try {
    found = do_single_exists(conn, batch->keys[idx]) ? 1 : 0;
  } catch (const std::exception& e) {
    failed = true;
    err = e.what();
  } catch (...) {
    failed = true;
    err = "unknown exception";
  }

  lk.lock();
  if (failed) {
    batch->any_failed = true;
    if (batch->first_error.empty()) {
      batch->first_error = std::move(err);
    }
  }
  finish_key(lk, *batch, idx, found);
}

void FSConnector::run_read(WorkerFSConn& conn,
                           std::unique_lock<std::mutex>& lk) {
  auto [batch, idx] = claim_next_key(get_queue_);
  const size_t len = batch->buf_lens[idx];
  read_bytes_in_flight_ += len;
  ++reads_in_flight_;
  unlock_and_wake_next(lk);

  uint8_t ok = 1;
  const std::string& key = batch->keys[idx];
  try {
    do_single_get(conn, key, batch->buf_ptrs[idx], len,
                  batch->batch_chunk_num_bytes);
  } catch (const std::exception& e) {
    ok = 0;
    fprintf(stderr, "[LMCache GET] key %s failed: %s\n", key.c_str(), e.what());
  } catch (...) {
    ok = 0;
    fprintf(stderr, "[LMCache GET] key %s failed: unknown exception\n",
            key.c_str());
  }

  lk.lock();
  read_bytes_in_flight_ -= len;
  --reads_in_flight_;
  finish_key(lk, *batch, idx, ok);
}

std::pair<std::shared_ptr<FSConnector::PoolBatch>, size_t>
FSConnector::claim_next_key(PoolQueue& queue) {
  std::shared_ptr<PoolBatch> batch = queue.front();
  const size_t idx = batch->next_idx++;
  if (batch->next_idx == batch->keys.size()) {
    queue.pop_front();
  }
  return {std::move(batch), idx};
}

void FSConnector::unlock_and_wake_next(std::unique_lock<std::mutex>& lk) {
  const bool more = has_ready_read_work();
  lk.unlock();
  if (more) {
    read_cv_.notify_one();
  }
}

void FSConnector::finish_key(std::unique_lock<std::mutex>& lk, PoolBatch& batch,
                             size_t idx, uint8_t ok) {
  batch.ok[idx] = ok;
  if (--batch.remaining != 0) {
    return;
  }
  Completion comp;
  comp.future_id = batch.future_id;
  comp.ok = !batch.any_failed;
  comp.error = std::move(batch.first_error);
  comp.result_bytes = std::move(batch.ok);
  unlock_and_wake_next(lk);
  push_completion(std::move(comp));
  lk.lock();
}

}  // namespace connector
}  // namespace lmcache
