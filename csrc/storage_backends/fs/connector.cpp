// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include <cerrno>
#include <cstdio>
#include <stdexcept>
#include <string>

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
  // Input key format (from _object_key_to_string — always @@-prefixed):
  //   @@<cache_salt>@<model_name>@<kv_rank_hex>@<chunk_hash_hex>
  //   (empty salt: @@@model_name@kv_rank_hex@chunk_hash_hex)
  //
  // Legacy keys without @@ prefix are still accepted for backward
  // compatibility but should no longer be produced by new code.
  //
  // Output filename always uses the @@-prefixed format:
  //   @@<cache_salt>@<model_name_safe>@0x<kv_rank_hex>@<chunk_hash_hex>.data
  //
  // NOTE: cache_salt must not contain '@', '/', '\', or NUL —
  // invariant enforced on the Python side.

  std::string cache_salt;
  std::string work_key = key;
  if (work_key.size() >= 2 && work_key[0] == KEY_SEP &&
      work_key[1] == KEY_SEP) {
    // @@-prefixed format: @@<salt>@<model>@<rank>@<hash>
    size_t salt_end = work_key.find(KEY_SEP, 2);
    if (salt_end == std::string::npos) {
      return key + FILE_EXT;
    }
    cache_salt = work_key.substr(2, salt_end - 2);
    work_key = work_key.substr(salt_end + 1);
  }
  // else: legacy key without @@ prefix — still accepted for backward
  // compatibility. The output filename will use the @@-prefixed format
  // regardless, so re-storing migrates the entry.

  // Remainder (or legacy input):
  //   <model_name>@<kv_rank_hex>@<chunk_hash_hex>
  // model_name may contain '@', so rsplit from the right.
  size_t last_sep = work_key.rfind(KEY_SEP);
  if (last_sep == std::string::npos) {
    return key + FILE_EXT;
  }
  size_t second_sep = work_key.rfind(KEY_SEP, last_sep - 1);
  if (second_sep == std::string::npos) {
    return key + FILE_EXT;
  }

  std::string model_name = work_key.substr(0, second_sep);
  std::string kv_rank_hex =
      work_key.substr(second_sep + 1, last_sep - second_sep - 1);
  std::string chunk_hash = work_key.substr(last_sep + 1);

  // Replace '/' with '-SEP-' for filesystem safety
  std::string safe_model = replace_all(model_name, "/", PATH_SLASH_REPLACEMENT);

  // Always emit @@-prefixed filename (even for empty salt / legacy input)
  // so all new files on disk use the unified format.
  std::string result;
  result.reserve(cache_salt.size() + safe_model.size() + kv_rank_hex.size() +
                 chunk_hash.size() + 32);
  // Leading @@ + salt + @
  result += KEY_SEP;
  result += KEY_SEP;
  result += cache_salt;
  result += KEY_SEP;
  result += safe_model;
  result += KEY_SEP;
  result += "0x";
  result += kv_rank_hex;
  result += KEY_SEP;
  result += chunk_hash;
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

// ---------------------------------------------------------------
// FSConnector
// ---------------------------------------------------------------

FSConnector::FSConnector(std::string base_path, int num_workers,
                         std::string relative_tmp_dir, bool use_odirect,
                         size_t read_ahead_size)
    : ConnectorBase(num_workers),
      base_path_(std::move(base_path)),
      relative_tmp_dir_(std::move(relative_tmp_dir)),
      use_odirect_(use_odirect),
      disk_block_size_(0),
      read_ahead_size_(read_ahead_size) {
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

  start_workers();  // IMPORTANT: call at END of constructor
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
  bool do_odirect = conn.use_odirect;
  if (do_odirect) {
    bool aligned = conn.disk_block_size > 0 && len % conn.disk_block_size == 0;
    if (aligned) {
#ifdef O_DIRECT
      flags |= O_DIRECT;
#endif
    } else {
      do_odirect = false;
    }
  }

  int fd = ::open(file_path.c_str(), flags);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + file_path.string() +
                             ": " + strerror(errno));
  }

  try {
    size_t n;
    if (conn.read_ahead_size > 0 && len > conn.read_ahead_size) {
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
  bool do_odirect = conn.use_odirect;
  if (do_odirect) {
    bool aligned = conn.disk_block_size > 0 && len % conn.disk_block_size == 0;
    if (aligned) {
#ifdef O_DIRECT
      flags |= O_DIRECT;
#endif
    } else {
      do_odirect = false;
    }
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

}  // namespace connector
}  // namespace lmcache
