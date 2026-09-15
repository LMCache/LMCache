// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
// Authors: Wenwen Chen <wenwen.chen@samsung.com>

#include "connector.h"
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <functional>
#include <limits.h>
#include <thread>
#include <chrono>

namespace lmcache {
namespace connector {

/**
 * Construct a new Hf3fsConnector.
 *
 * Validates:
 * - mount_point is a valid 3FS path (via hf3fs_extract_mount_point)
 * - Each base_path is a subdirectory of mount_point
 * - Parameters are within valid ranges
 */
Hf3fsConnector::Hf3fsConnector(std::string mount_point, std::string base_paths,
                               int num_workers, int ior_entries, int io_depth,
                               int numa_id, size_t iov_size, int time_out,
                               bool enable_key_buffer,
                               WorkerPoolConfig worker_pool_config)
    : ConnectorBase(num_workers, std::move(worker_pool_config)),
      mount_point_(std::move(mount_point)),
      ior_entries_(ior_entries),
      io_depth_(io_depth),
      numa_id_(numa_id),
      iov_size_(iov_size),
      time_out_(time_out),
      buffer_enabled_(enable_key_buffer) {
  // Debug print worker pool configuration
  fprintf(stderr,
          "[LMCache HF3FS] Worker pool: num_workers=%d, "
          "per_op_workers={",
          num_workers_);
  for (const auto& [key, count] : worker_pool_config_.per_op_workers) {
    fprintf(stderr, "%s=\"%d\"", key.c_str(), count);
  }
  fprintf(stderr, "}\n");

  // Validate mount_point using 3FS SDK
  char mount_point_buf[PATH_MAX];
  int ret = hf3fs_extract_mount_point(mount_point_buf, sizeof(mount_point_buf),
                                      mount_point_.c_str());
  if (ret < 0) {
    throw std::runtime_error("hf3fs_extract_mount_point failed: '" +
                             mount_point_ + "' is not a valid 3FS path");
  }
  if (ret > static_cast<int>(sizeof(mount_point_buf))) {
    throw std::runtime_error("Mount point path too long: '" + mount_point_ +
                             "'");
  }

  std::string extracted(mount_point_buf);
  if (extracted != mount_point_) {
    throw std::runtime_error("Mount point mismatch: config='" + mount_point_ +
                             "', extracted='" + extracted + "'");
  }

  // Parse comma-separated base_paths
  std::stringstream ss(base_paths);
  std::string path;
  while (std::getline(ss, path, ',')) {
    // Trim whitespace
    path.erase(0, path.find_first_not_of(" \t"));
    path.erase(path.find_last_not_of(" \t") + 1);
    if (!path.empty()) {
      base_paths_.push_back(path);
    }
  }

  if (base_paths_.empty()) {
    throw std::runtime_error("No valid base_paths provided");
  }

  // Validate each base_path is a subdirectory of mount_point
  for (const auto& bp : base_paths_) {
    if (bp.find(mount_point_) != 0) {
      throw std::runtime_error("base_path '" + bp +
                               "' is not a subdirectory of mount_point '" +
                               mount_point_ + "'");
    }
  }

  if (buffer_enabled_) {
    fprintf(stderr, "[LMCache HF3FS] Buffer enabled, begin scan...\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    scan_and_build_buffer_();
    auto t1 = std::chrono::high_resolution_clock::now();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    fprintf(stderr,
            "[LMCache HF3FS] End of scan, scanned %zu keys from %zu "
            "base_path(s), elapsed %ld ms\n",
            key_buffer_.size(), base_paths_.size(), elapsed_ms);
  } else {
    fprintf(stderr, "[LMCache HF3FS] Buffer disabled\n");
  }

  // IMPORTANT: start worker threads at the END of the constructor
  start_workers();
}

/**
 * Initialize read Ior (I/O ring).
 *
 * @param ior Ior structure to initialize
 * @throws std::runtime_error if hf3fs_iorcreate4 fails
 */
void Hf3fsConnector::init_read_ior(hf3fs_ior& ior) {
  int ret = hf3fs_iorcreate4(&ior, mount_point_.c_str(), ior_entries_,
                             true,  // for_read=true
                             io_depth_,
                             time_out_,  // timeout (ms)
                             numa_id_,
                             0);  // flags
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iorcreate4 (read) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize write Ior (I/O ring).
 *
 * @param ior Ior structure to initialize
 * @throws std::runtime_error if hf3fs_iorcreate4 fails
 */
void Hf3fsConnector::init_write_ior(hf3fs_ior& ior) {
  int ret = hf3fs_iorcreate4(&ior, mount_point_.c_str(), ior_entries_,
                             false,  // for_read=false
                             io_depth_,
                             time_out_,  // timeout (ms)
                             numa_id_,
                             0);  // flags
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iorcreate4 (write) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize read Iov (I/O buffer).
 *
 * @param iov Iov structure to initialize
 * @throws std::runtime_error if hf3fs_iovcreate fails
 */
void Hf3fsConnector::init_read_iov(hf3fs_iov& iov) {
  int ret = hf3fs_iovcreate(&iov, mount_point_.c_str(), iov_size_,
                            0,  // block_size
                            numa_id_);
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iovcreate (read) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Initialize write Iov (I/O buffer).
 *
 * @param iov Iov structure to initialize
 * @throws std::runtime_error if hf3fs_iovcreate fails
 */
void Hf3fsConnector::init_write_iov(hf3fs_iov& iov) {
  int ret = hf3fs_iovcreate(&iov, mount_point_.c_str(), iov_size_,
                            0,  // block_size
                            numa_id_);
  if (ret < 0) {
    throw std::runtime_error("hf3fs_iovcreate (write) failed: " +
                             std::to_string(-ret));
  }
}

/**
 * Create a new per-thread connection.
 *
 * Initializes:
 * - Read Ior and Iov
 * - Write Ior and Iov
 * - File descriptors (initialized to -1)
 *
 * @return WorkerHf3fsConn Initialized connection structure
 */
WorkerHf3fsConn Hf3fsConnector::create_connection() {
  WorkerHf3fsConn conn;

  // Initialize read Ior
  init_read_ior(conn.read_ior);
  conn.read_ior_initialized = true;

  // Initialize write Ior
  init_write_ior(conn.write_ior);
  conn.write_ior_initialized = true;

  // Initialize read Iov
  init_read_iov(conn.read_iov);
  conn.read_iov_initialized = true;

  // Initialize write Iov
  init_write_iov(conn.write_iov);
  conn.write_iov_initialized = true;

  // File descriptors start uninitialized
  conn.fd = -1;

  return conn;
}

/**
 * Open a file for read or write.
 *
 * Steps:
 * 1. Open file with appropriate flags (O_RDONLY or O_WRONLY|O_CREAT)
 * 2. Register FD with 3FS via hf3fs_reg_fd
 * 3. Store file path in connection
 *
 * @param conn Connection to use
 * @param file_path Path to file
 * @param for_write true for write, false for read
 * @throws std::runtime_error if open or hf3fs_reg_fd fails
 */
void Hf3fsConnector::open_file(WorkerHf3fsConn& conn,
                               const std::string& file_path, bool for_write) {
  int flags = for_write ? (O_WRONLY | O_CREAT) : O_RDONLY;
  mode_t mode = for_write ? 0644 : 0;

  conn.fd = ::open(file_path.c_str(), flags, mode);
  if (conn.fd < 0) {
    throw std::runtime_error("open failed: " + std::string(strerror(errno)));
  }

  int reg_result = hf3fs_reg_fd(conn.fd, 0);
  if (reg_result > 0) {
    ::close(conn.fd);
    conn.fd = -1;
    throw std::runtime_error(
        "hf3fs_reg_fd failed: " + std::to_string(reg_result) +
        " (errno=" + strerror(reg_result) + ")");
  }
  conn.registered = true;
  conn.file_path = file_path;
}

/**
 * Close a file and deregister from 3FS.
 *
 * Steps:
 * 1. Deregister FD from 3FS via hf3fs_dereg_fd
 * 2. Close Linux FD
 * 3. Clear file path
 *
 * @param conn Connection to use
 */
void Hf3fsConnector::close_file(WorkerHf3fsConn& conn) {
  // Step 1: Deregister FD from 3FS first (only if registration succeeded)
  if (conn.registered) {
    hf3fs_dereg_fd(conn.fd);
    conn.registered = false;
  }
  // Step 2: Close Linux FD
  if (conn.fd >= 0) {
    ::close(conn.fd);
    conn.fd = -1;
  }
  conn.file_path.clear();
}

/**
 * Read data from a file using 3FS I/O.
 *
 * Steps (per chunk):
 * 1. Prepare I/O request via hf3fs_prep_io
 * 2. Submit I/O via hf3fs_submit_ios
 * 3. Wait for completion via hf3fs_wait_for_ios
 * 4. Copy data from Iov buffer to output buffer
 *
 * If len > iov_size_, data is read in multiple chunks.
 *
 * @param conn Connection to use
 * @param ior Ior to use for I/O
 * @param iov Iov buffer for data
 * @param buf Output buffer
 * @param len Number of bytes to read
 * @throws std::runtime_error if any 3FS operation fails
 */
void Hf3fsConnector::read_file(WorkerHf3fsConn& conn, hf3fs_ior& ior,
                               hf3fs_iov& iov, void* buf, size_t len) {
  size_t file_offset = 0;
  size_t total_read = 0;

  while (file_offset < len) {
    size_t sub_len = std::min(iov_size_, len - file_offset);

    int ret = hf3fs_prep_io(&ior, &iov, true, iov.base, conn.fd, file_offset,
                            sub_len, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_prep_io failed: " + std::to_string(-ret));
    }

    ret = hf3fs_submit_ios(&ior);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_submit_ios failed: " +
                               std::to_string(-ret));
    }

    hf3fs_cqe cqe;
    ret = hf3fs_wait_for_ios(&ior, &cqe, 1, 1, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_wait_for_ios failed: " +
                               std::to_string(-ret));
    }
    if (cqe.result < 0) {
      throw std::runtime_error("I/O failed: " + std::to_string(-cqe.result));
    }

    memcpy(static_cast<char*>(buf) + total_read, iov.base, sub_len);
    total_read += sub_len;
    file_offset += sub_len;
  }
}

/**
 * Write data to a file using 3FS I/O.
 *
 * Steps (per chunk):
 * 1. Copy data to Iov buffer
 * 2. Prepare I/O request via hf3fs_prep_io
 * 3. Submit I/O via hf3fs_submit_ios
 * 4. Wait for completion via hf3fs_wait_for_ios
 *
 * If len > iov_size_, data is written in multiple chunks.
 *
 * @param conn Connection to use
 * @param ior Ior to use for I/O
 * @param iov Iov buffer for data
 * @param buf Input buffer
 * @param len Number of bytes to write
 * @throws std::runtime_error if any 3FS operation fails
 */
void Hf3fsConnector::write_file(WorkerHf3fsConn& conn, hf3fs_ior& ior,
                                hf3fs_iov& iov, const void* buf, size_t len) {
  size_t file_offset = 0;
  size_t total_written = 0;

  while (file_offset < len) {
    size_t sub_len = std::min(iov_size_, len - file_offset);

    // Copy data to Iov buffer
    memcpy(iov.base, static_cast<const char*>(buf) + file_offset, sub_len);

    int ret = hf3fs_prep_io(&ior, &iov, false, iov.base, conn.fd, file_offset,
                            sub_len, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_prep_io failed: " + std::to_string(-ret));
    }

    ret = hf3fs_submit_ios(&ior);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_submit_ios failed: " +
                               std::to_string(-ret));
    }

    hf3fs_cqe cqe;
    ret = hf3fs_wait_for_ios(&ior, &cqe, 1, 1, nullptr);
    if (ret < 0) {
      throw std::runtime_error("hf3fs_wait_for_ios failed: " +
                               std::to_string(-ret));
    }
    if (cqe.result < 0) {
      throw std::runtime_error("I/O failed: " + std::to_string(-cqe.result));
    }

    total_written += sub_len;
    file_offset += sub_len;
  }
}

/**
 * Convert a key string to a filesystem-safe filename.
 *
 *   Unsalted:  <safe_model>@0x<kv_rank_hex>@<chunk_hash_hex>.data
 *   Salted  :  <safe_model>@0x<kv_rank_hex>@<chunk_hash_hex>@<cache_salt>.data
 *
 * Input key format: "{model}@{kv_rank_hex}@{chunk_hash_hex}@{cache_salt?}"
 *
 * Transformations:
 * - Split on '@' — must yield 3 (unsalted) or 4 (salted) fields.
 * - Replace '/' in model_name only with '-SEP-' (not in kv_rank or hash).
 * - Add '0x' prefix to kv_rank_hex.
 * - Append '.data' extension.
 *
 * NOTE: both model_name and cache_salt are forbidden from containing
 * '@' (invariant enforced on the Python side), so splitting on '@'
 * is unambiguous.
 *
 * @param key Key string from Python layer
 * @return Filesystem-safe filename
 */
std::string Hf3fsConnector::key_to_filename(const std::string& key) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == '@') {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 3 && parts.size() != 4) {
    fprintf(stderr,
            "[LMCache HF3FS] key_to_filename: malformed key "
            "(expected 3 or 4 '@'-separated fields): %s\n",
            key.c_str());
    throw std::runtime_error(
        "Hf3fsConnector: malformed key (expected 3 or 4 '@'-separated "
        "fields): " +
        key);
  }

  const std::string& model_name = parts[0];
  const std::string& kv_rank_hex = parts[1];
  const std::string& chunk_hash = parts[2];
  const std::string cache_salt = parts.size() == 4 ? parts[3] : std::string();

  // Replace '/' with '-SEP-' only in model_name for filesystem safety
  std::string safe_model = model_name;
  size_t spos = 0;
  while ((spos = safe_model.find('/', spos)) != std::string::npos) {
    safe_model.replace(spos, 1, "-SEP-");
    spos += 5;
  }

  // Build filename with '0x' prefix on kv_rank (matches FSConnector and
  // Python-side _object_key_to_filename).
  std::string result;
  result.reserve(safe_model.size() + kv_rank_hex.size() + chunk_hash.size() +
                 cache_salt.size() + 32);
  result += safe_model;
  result += '@';
  result += "0x";
  result += kv_rank_hex;
  result += '@';
  result += chunk_hash;
  if (!cache_salt.empty()) {
    result += '@';
    result += cache_salt;
  }
  result += ".data";
  return result;
}

/**
 * Select a base path based on the key's chunk_hash.
 *
 * Algorithm:
 * 1. Extract chunk_hash from key (format: "model@kv_rank@chunk_hash@salt")
 * 2. Take first 16 characters of hash (64 bits)
 * 3. Convert to uint64_t and mod by number of paths
 *
 * This provides hash-based distribution for load balancing.
 *
 * @param key Key string
 * @return Selected base path
 */
const std::string& Hf3fsConnector::select_base_path(
    const std::string& key) const {
  if (base_paths_.size() == 1) {
    return base_paths_[0];
  }

  // Extract chunk_hash from key
  // Format: "{model}@{kv_rank_hex}@{chunk_hash_hex}@{cache_salt?}"
  size_t first_at = key.find('@');
  if (first_at == std::string::npos) {
    return base_paths_[0];
  }
  size_t second_at = key.find('@', first_at + 1);
  if (second_at == std::string::npos) {
    return base_paths_[0];
  }
  size_t hash_start = second_at + 1;
  size_t hash_end = key.find('@', hash_start);
  if (hash_end == std::string::npos) {
    hash_end = key.length();
  }

  // Extract full hash string
  std::string hash_str = key.substr(hash_start, hash_end - hash_start);

  // Use last 16 characters (64 bits) for better distribution
  // Sequential keys like 00000000000000000000000000000000,
  // 00000000000000000000000000000001 have zeros in first 16 chars
  // but varying values in last 16 chars
  size_t hash_len = hash_str.length();
  size_t substr_start = (hash_len > 16) ? (hash_len - 16) : 0;
  std::string hash_substr = hash_str.substr(substr_start);

  // Convert to uint64_t and select path
  uint64_t hash_val = std::stoull(hash_substr, nullptr, 16);
  size_t idx = hash_val % base_paths_.size();

  return base_paths_[idx];
}

/**
 * Convert a key to a full file path.
 *
 * @param key Key string
 * @return Full file path (base_path + filename)
 */
std::string Hf3fsConnector::key_to_path(const std::string& key) {
  const std::string& base_path = select_base_path(key);
  std::string filename = key_to_filename(key);
  return base_path + "/" + filename;
}

/**
 * Retrieve data for a key.
 *
 * Steps:
 * 1. Convert key to file path
 * 2. Open file for reading
 * 3. Read data using 3FS I/O
 * 4. Close file
 *
 * @param conn Connection to use
 * @param key Key to retrieve
 * @param buf Output buffer
 * @param len Number of bytes to read
 * @param chunk_size Chunk size (unused for 3FS)
 */
void Hf3fsConnector::do_single_get(WorkerHf3fsConn& conn,
                                   const std::string& key, void* buf,
                                   size_t len, size_t chunk_size) {
  (void)chunk_size;  // Unused for 3FS
  std::string file_path = key_to_path(key);
  open_file(conn, file_path, false);
  read_file(conn, conn.read_ior, conn.read_iov, buf, len);
  close_file(conn);
}

/**
 * Store data for a key.
 *
 * Steps:
 * 1. Convert key to file path
 * 2. Open file for writing
 * 3. Write data using 3FS I/O
 * 4. Close file
 *
 * @param conn Connection to use
 * @param key Key to store
 * @param buf Input buffer
 * @param len Number of bytes to write
 * @param chunk_size Chunk size (unused for 3FS)
 */
void Hf3fsConnector::do_single_set(WorkerHf3fsConn& conn,
                                   const std::string& key, const void* buf,
                                   size_t len, size_t chunk_size) {
  (void)chunk_size;  // Unused for 3FS

  // Skip if already stored on disk
  // if (do_single_exists(conn, key)) {
  //  return;
  // }

  std::string file_path = key_to_path(key);
  open_file(conn, file_path, true);
  write_file(conn, conn.write_ior, conn.write_iov, buf, len);
  close_file(conn);
  if (buffer_enabled_) {
    buffer_add_(key);
  }
}

/**
 * Check if a key exists.
 *
 * @param conn Connection to use
 * @param key Key to check
 * @return true if key exists, false otherwise
 */
bool Hf3fsConnector::do_single_exists(WorkerHf3fsConn& conn,
                                      const std::string& key) {
  (void)conn;  // Unused
  if (!buffer_enabled_) {
    std::string file_path = key_to_path(key);
    return std::filesystem::exists(file_path);
  }
  return buffer_contains_(key);
}

/**
 * Delete a key.
 *
 * @param conn Connection to use
 * @param key Key to delete
 * @return true if deleted, false if not found
 */
bool Hf3fsConnector::do_single_delete(WorkerHf3fsConn& conn,
                                      const std::string& key) {
  (void)conn;  // Unused
  std::string file_path = key_to_path(key);
  try {
    bool removed = std::filesystem::remove(file_path);
    if (removed && buffer_enabled_) {
      buffer_remove_(key);
    }
    return removed;
  } catch (const std::filesystem::filesystem_error& e) {
    (void)e;  // Suppress unused warning
    return false;
  }
}

/**
 * Scan all base_paths in parallel and populate the key buffer.
 *
 * Runs one thread per base_path (bounded by num_workers_), each thread
 * builds a private local_set to avoid contention, then merges into
 * key_buffer_ in a single-threaded pass.
 *
 * Called from the constructor (single-threaded, before workers start).
 */
void Hf3fsConnector::scan_and_build_buffer_() {
  if (base_paths_.empty()) {
    return;
  }

  int num_workers =
      std::min(num_workers_, static_cast<int>(base_paths_.size()));
  if (num_workers <= 0) {
    num_workers = 1;
  }

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(num_workers));
  std::vector<std::unordered_set<std::string>> local_sets(num_workers);

  for (int i = 0; i < num_workers; ++i) {
    size_t start = static_cast<size_t>(i) * (base_paths_.size() / num_workers);
    size_t end =
        (i == num_workers - 1)
            ? base_paths_.size()
            : static_cast<size_t>(i + 1) * (base_paths_.size() / num_workers);
    if (start >= end) {
      break;
    }

    threads.emplace_back(&Hf3fsConnector::buffer_scan_worker_, this,
                         base_paths_[start], std::ref(local_sets[i]));
  }

  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  // Single-threaded merge -- no contention
  for (int i = 0; i < num_workers; ++i) {
    for (const auto& key : local_sets[i]) {
      key_buffer_.Insert(key);
    }
  }
}

/**
 * Scan one base_path directory and recover keys from .data filenames.
 *
 * Inverts key_to_filename() for the new format:
 *   Filename: <safe_model>@0x<kv_rank_hex>@<chunk_hash_hex>.data
 *   Key:      <model>@<kv_rank_hex>@<chunk_hash_hex>
 *
 * Steps:
 * 1. Skip non-regular files (e.g. subdirectories)
 * 2. Skip files that do not end with ".data"
 * 3. Strip the ".data" suffix
 * 4. Split on '@' and remove '0x' prefix from kv_rank to recover original key
 * 5. Replace '-SEP-' with '/' to recover the original model_name
 * 6. Insert the recovered key into the provided local set
 *
 * Called from worker threads spawned in scan_and_build_buffer_().
 *
 * @param base_path Directory to scan for .data files
 * @param local_set Set to populate with recovered keys
 */
void Hf3fsConnector::buffer_scan_worker_(
    const std::string& base_path, std::unordered_set<std::string>& local_set) {
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(base_path, ec)) {
    if (!entry.is_regular_file(ec)) {
      continue;
    }

    const std::string filename = entry.path().filename().string();
    if (filename.length() <= 5 ||
        filename.compare(filename.length() - 5, 5, ".data") != 0) {
      continue;
    }

    std::string key = filename.substr(0, filename.length() - 5);

    // Split on '@' to recover original key parts
    // Format: <safe_model>@0x<kv_rank_hex>@<chunk_hash_hex>@<cache_salt?>
    std::vector<std::string> parts;
    size_t start = 0;
    for (size_t pos = 0; pos <= key.size(); ++pos) {
      if (pos == key.size() || key[pos] == '@') {
        parts.emplace_back(key.substr(start, pos - start));
        start = pos + 1;
      }
    }

    if (parts.size() < 3) {
      // Malformed filename, skip
      continue;
    }

    // Reconstruct key: <model>@<kv_rank_hex>@<chunk_hash_hex>[@<cache_salt>]
    // Remove "0x" prefix from kv_rank field (parts[1])
    std::string kv_rank = parts[1];
    if (kv_rank.size() >= 2 && kv_rank.substr(0, 2) == "0x") {
      kv_rank = kv_rank.substr(2);
    }

    std::string recovered;
    recovered += parts[0];
    recovered += '@';
    recovered += kv_rank;
    recovered += '@';
    recovered += parts[2];
    for (size_t i = 3; i < parts.size(); ++i) {
      recovered += '@';
      recovered += parts[i];
    }

    // Replace '-SEP-' with '/' to recover original model_name with slashes
    size_t spos = 0;
    while ((spos = recovered.find("-SEP-", spos)) != std::string::npos) {
      recovered.replace(spos, 5, "/");
      spos += 1;
    }

    local_set.insert(std::move(recovered));
  }
}

/**
 * Add a key to the buffer.
 *
 * Thread-safe via concurrent_flat_hash_set (sharded locks).
 * Called after a successful do_single_set().
 */
void Hf3fsConnector::buffer_add_(const std::string& key) {
  key_buffer_.Insert(key);
}

/**
 * Remove a key from the buffer.
 *
 * Thread-safe via concurrent_flat_hash_set (sharded locks).
 * Called after a successful do_single_delete().
 */
void Hf3fsConnector::buffer_remove_(const std::string& key) {
  key_buffer_.Erase(key);
}

/**
 * Check if a key exists in the buffer.
 *
 * Thread-safe read using sharded_flat_hash_set::Contains().
 * Each shard has its own lock, reducing contention.
 *
 * @param key Key to check
 * @return true if key is in the buffer, false otherwise
 */
bool Hf3fsConnector::buffer_contains_(const std::string& key) const {
  return key_buffer_.Contains(key);
}

/**
 * Optimized batch exists using the in-memory key buffer.
 *
 * When the buffer is enabled, all lookups are answered from the hash
 * set in microseconds. When disabled, falls back to the base class
 * default which iterates do_single_exists (filesystem).
 */
void Hf3fsConnector::do_batch_exists(WorkerHf3fsConn& conn,
                                     const Request& req) {
  (void)conn;
  if (!buffer_enabled_) {
    ConnectorBase::do_batch_exists(conn, req);
    return;
  }
  // Buffer path: check cache for all keys
  for (size_t i = 0; i < req.keys.size(); ++i) {
    req.batch->per_key_results[req.start_idx + i] =
        buffer_contains_(req.keys[i]) ? 1 : 0;
  }
}

/**
 * Shutdown all connections.
 *
 * Worker destructors handle cleanup automatically,
 * so this method is a no-op.
 */
void Hf3fsConnector::shutdown_connections() {
  // Worker destructors handle cleanup
}

}  // namespace connector
}  // namespace lmcache
