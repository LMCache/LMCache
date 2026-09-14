// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace lmcache {
namespace connector {

// ---------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------

std::string FSConnector::replace_all(const std::string& str,
                                     const std::string& from,
                                     const std::string& to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.size(), to);
    pos += to.size();
  }
  return result;
}

std::string FSConnector::key_to_filename(const std::string& key) {
  // Input key format (from _object_key_to_string):
  //   Unsalted: <model_name>@<kv_rank_hex>@<chunk_hash_hex>
  //   Salted  : <model_name>@<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>
  //
  // Output filename (matching fs_l2_adapter.py._object_key_to_filename):
  //   Unsalted: <model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>.data
  //   Salted  :
  //   <model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>.data
  //
  // The unsalted 3-field shape is bit-identical to the pre-cache_salt
  // format, so existing cache directories remain valid.
  //
  // NOTE: both model_name and cache_salt are forbidden from containing
  // '@' (invariant enforced on the Python side), so splitting on '@'
  // is unambiguous — no marker, no rsplit.

  // Split on '@' — must yield 3 (unsalted) or 4 (salted) fields.
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 3 && parts.size() != 4) {
    throw std::runtime_error(
        "FSConnector: malformed key (expected 3 or 4 '@'-separated fields): " +
        key);
  }

  const std::string& model_name = parts[0];
  const std::string& kv_rank_hex = parts[1];
  const std::string& chunk_hash = parts[2];
  const std::string cache_salt = parts.size() == 4 ? parts[3] : std::string();

  // Replace '/' with '-SEP-' for filesystem safety
  std::string safe_model = replace_all(model_name, "/", PATH_SLASH_REPLACEMENT);

  // Emit filename. Salt is appended at the tail so the unsalted shape
  // matches what older builds wrote to disk.
  std::string result;
  result.reserve(safe_model.size() + kv_rank_hex.size() + chunk_hash.size() +
                 cache_salt.size() + 32);
  result += safe_model;
  result += KEY_SEP;
  result += "0x";
  result += kv_rank_hex;
  result += KEY_SEP;
  result += chunk_hash;
  if (!cache_salt.empty()) {
    result += KEY_SEP;
    result += cache_salt;
  }
  result += FILE_EXT;
  return result;
}

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
static size_t effective_read_budget(int depth, size_t budget) {
  if (depth <= 0) {
    return 0;
  }
  return budget == 0 ? kDefaultReadMaxBytesInFlight : budget;
}

FSConnector::FSConnector(std::string base_path, int num_workers,
                         std::string relative_tmp_dir, bool use_odirect,
                         size_t read_ahead_size, int read_io_depth,
                         size_t read_max_bytes_in_flight)
    : ConnectorBase(num_workers),
      base_path_(std::move(base_path)),
      relative_tmp_dir_(std::move(relative_tmp_dir)),
      use_odirect_(use_odirect),
      disk_block_size_(0),
      read_ahead_size_(read_ahead_size),
      read_max_bytes_in_flight_(
          effective_read_budget(read_io_depth, read_max_bytes_in_flight)) {
  if (read_io_depth < 0) {
    throw std::runtime_error("read_io_depth must be >= 0");
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
    start_read_pool(read_io_depth);
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

bool FSConnector::do_single_delete(WorkerFSConn& conn, const std::string& key) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;
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
  // See connector.h.  A large batch is still spread over the workers so
  // their per-worker budget shares add up to the whole budget.
  const size_t depth = read_threads_.size();
  const size_t tiles_for_depth = (num_items + depth - 1) / depth;
  const size_t workers = static_cast<size_t>(worker_count_for_op(op));
  return std::max<size_t>(1, std::min(workers, tiles_for_depth));
}

void FSConnector::do_batch_get(WorkerFSConn& conn, const Request& req) {
  if (read_threads_.empty()) {
    ConnectorBase::do_batch_get(conn, req);
    return;
  }
  do_batch_get_pooled(req);
}

void FSConnector::on_workers_stopped() { stop_read_pool(); }

void FSConnector::do_batch_get_pooled(const Request& req) {
  const size_t num_keys = req.keys.size();

  // Bounded by the reader count and by the byte budget, whichever binds.
  const size_t max_count = read_threads_.size();
  const size_t budget = worker_read_budget_bytes();

  for (size_t base = 0; base < num_keys;) {
    size_t end = base;
    size_t group_bytes = 0;
    while (end < num_keys && end - base < max_count) {
      const size_t len = req.buf_lens[end];
      // The first object is always taken, or one larger than the budget
      // could never be read.
      if (end > base && group_bytes + len > budget) {
        break;
      }
      group_bytes += len;
      ++end;
    }
    const size_t count = end - base;

    ReadGroup group;
    group.ok.assign(count, 1);
    group.remaining = count;

    // Built off to the side so an allocation failure cannot leave a
    // half-published group behind.
    std::list<ReadTask> pending;
    for (size_t local = 0; local < count; ++local) {
      pending.push_back(ReadTask{&req, base + local, local, &group});
    }

    {
      std::unique_lock<std::mutex> lk(read_mu_);
      read_queue_.splice(read_queue_.end(), pending);
      read_cv_.notify_all();
      read_done_cv_.wait(lk, [&group] { return group.remaining == 0; });
    }

    for (size_t local = 0; local < count; ++local) {
      req.batch->per_key_results[req.start_idx + base + local] =
          group.ok[local];
    }
    base = end;
  }
}

size_t FSConnector::worker_read_budget_bytes() const {
  const size_t workers =
      static_cast<size_t>(worker_count_for_op(Op::BATCH_TILE_GET));
  return std::max<size_t>(1, read_max_bytes_in_flight_ / workers);
}

// Connections are built here so a failure surfaces in the constructor.
void FSConnector::start_read_pool(int read_io_depth) {
  if (read_io_depth <= 0) {
    return;
  }
  const size_t depth = static_cast<size_t>(read_io_depth);
  read_conns_.reserve(depth);
  for (size_t i = 0; i < depth; ++i) {
    read_conns_.push_back(create_connection());
  }
  read_threads_.reserve(depth);
  for (size_t i = 0; i < depth; ++i) {
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
  for (;;) {
    ReadTask task;
    {
      std::unique_lock<std::mutex> lk(read_mu_);
      read_cv_.wait(lk, [this] { return read_stop_ || !read_queue_.empty(); });
      if (read_queue_.empty()) {
        return;  // the predicate admits an empty queue only on stop
      }
      task = read_queue_.front();
      read_queue_.pop_front();
    }

    uint8_t ok = 1;
    const std::string& key = task.req->keys[task.index];
    try {
      do_single_get(conn, key, task.req->buf_ptrs[task.index],
                    task.req->buf_lens[task.index],
                    task.req->batch_chunk_num_bytes);
    } catch (const std::exception& e) {
      ok = 0;
      fprintf(stderr, "[LMCache GET] key %s failed: %s\n", key.c_str(),
              e.what());
    } catch (...) {
      ok = 0;
      fprintf(stderr, "[LMCache GET] key %s failed: unknown exception\n",
              key.c_str());
    }

    {
      std::lock_guard<std::mutex> lk(read_mu_);
      ReadGroup& group = *task.group;
      group.ok[task.local] = ok;
      if (--group.remaining == 0) {
        read_done_cv_.notify_all();
      }
    }
  }
}

}  // namespace connector
}  // namespace lmcache
