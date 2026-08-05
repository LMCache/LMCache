// SPDX-License-Identifier: Apache-2.0

#include "connector.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace lmcache {
namespace connector {

namespace {

// Carries the errno alongside the message. A caller cannot classify an
// O_DIRECT refusal from strerror text alone.
struct IoError : std::runtime_error {
  IoError(const std::string& what, int errnum)
      : std::runtime_error(what), err(errnum) {}
  int err;
};

// Errnos meaning this file or filesystem will not serve direct I/O.
// open() is documented to answer EINVAL, a filesystem may answer
// EOPNOTSUPP, and EPERM is reachable under policy. None of them says the
// data is unavailable, so each should degrade to buffered I/O rather than
// fail the request.
bool is_direct_io_refusal(int err) {
  if (err == EINVAL || err == EPERM) return true;
#ifdef EOPNOTSUPP
  if (err == EOPNOTSUPP) return true;
#endif
#ifdef ENOTSUP
  if (err == ENOTSUP) return true;
#endif
  return false;
}

// O_DIRECT constrains the file offset, the transfer length and the buffer
// address. Every transfer here starts at offset zero, so the other two are
// checked. Length alone is not enough, because a pinned host allocator can
// return a block sized region at an address that is not block aligned.
//
// block_size reaches here as statvfs f_bsize, which is the filesystem's block
// size and not the device logical_block_size the kernel actually enforces for
// O_DIRECT. On mainstream setups f_bsize is a multiple of that sector size, so
// this check is the conservative side and anything it accepts the kernel
// accepts too. The gap only bites in the other direction, when choosing a
// deliberately misaligned address for a test: a buffer off an f_bsize boundary
// can still be a legal O_DIRECT address on a device with a smaller sector, and
// the O_DIRECT regression test passed with and without the fix for exactly
// that reason until its skew was changed.
bool odirect_alignment_ok(size_t block_size, size_t len, const void* buf) {
  if (block_size == 0) return false;
  if (len % block_size != 0) return false;
  return reinterpret_cast<uintptr_t>(buf) % block_size == 0;
}

// Opens with O_DIRECT when asked, and falls back to a buffered open when the
// flag is refused. Reports which one the caller got, since that decides
// whether the transfer has to stay aligned.
int open_maybe_direct(const char* path, int flags, mode_t mode,
                      bool want_direct, bool* opened_direct) {
  *opened_direct = false;
#ifdef O_DIRECT
  if (want_direct) {
    int fd = ::open(path, flags | O_DIRECT, mode);
    if (fd >= 0) {
      *opened_direct = true;
      return fd;
    }
    if (!is_direct_io_refusal(errno)) {
      return -1;
    }
  }
#else
  (void)want_direct;
#endif
  return ::open(path, flags, mode);
}

bool odirect_already_refused(const WorkerFSConn& conn) {
  return conn.odirect_refused != nullptr &&
         conn.odirect_refused->load(std::memory_order_relaxed);
}

void note_odirect_refused(WorkerFSConn& conn) {
  if (conn.odirect_refused == nullptr) return;
  if (!conn.odirect_refused->exchange(true, std::memory_order_relaxed)) {
    fprintf(stderr,
            "[LMCache FS] O_DIRECT refused on this path, continuing with "
            "buffered I/O\n");
  }
}

}  // namespace

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
      throw IoError("write failed: " + std::string(strerror(errno)), errno);
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
      throw IoError("read failed: " + std::string(strerror(errno)), errno);
    }
    if (n == 0) break;  // EOF
    total += static_cast<size_t>(n);
  }
  return total;
}

// Reads exactly len bytes, optionally issuing a small leading read first so
// the filesystem starts reading ahead. Shared by the direct and the buffered
// attempt so both paths behave identically.
static void read_whole_file(int fd, void* buf, size_t len,
                            size_t read_ahead_size,
                            const std::string& file_path) {
  size_t n;
  if (read_ahead_size > 0 && len > read_ahead_size) {
    size_t n_head = read_all(fd, buf, read_ahead_size);
    if (n_head < read_ahead_size) {
      n = n_head;  // short read on the head, treat as incomplete
    } else {
      size_t n_tail =
          read_all(fd, static_cast<char*>(buf) + read_ahead_size,
                   len - read_ahead_size);
      n = n_head + n_tail;
    }
  } else {
    n = read_all(fd, buf, len);
  }
  if (n != len) {
    throw std::runtime_error("incomplete read for " + file_path + ": expected " +
                             std::to_string(len) + ", got " +
                             std::to_string(n));
  }
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
  conn.odirect_refused = &odirect_refused_;
  return conn;
}

void FSConnector::do_single_get(WorkerFSConn& conn, const std::string& key,
                                void* buf, size_t len, size_t chunk_size) {
  std::string filename = key_to_filename(key);
  auto file_path = conn.base_path / filename;

  const int flags = O_RDONLY;
  bool want_odirect =
      conn.use_odirect && !odirect_already_refused(conn) &&
      odirect_alignment_ok(conn.disk_block_size, len, buf);

  bool opened_direct = false;
  int fd = open_maybe_direct(file_path.c_str(), flags, 0, want_odirect,
                             &opened_direct);
  if (fd < 0) {
    throw std::runtime_error("open for read failed: " + file_path.string() +
                             ": " + strerror(errno));
  }
  if (want_odirect && !opened_direct) {
    note_odirect_refused(conn);
  }

  try {
    read_whole_file(fd, buf, len, conn.read_ahead_size, file_path.string());
  } catch (const IoError& e) {
    ::close(fd);
    // O_DIRECT accepted at open() and refused at read() is reachable on
    // mainstream filesystems, so retry buffered once before failing the key.
    if (!opened_direct || !is_direct_io_refusal(e.err)) {
      throw;
    }
    note_odirect_refused(conn);
    fd = ::open(file_path.c_str(), flags);
    if (fd < 0) {
      throw std::runtime_error("open for buffered read failed: " +
                               file_path.string() + ": " + strerror(errno));
    }
    try {
      read_whole_file(fd, buf, len, conn.read_ahead_size, file_path.string());
    } catch (...) {
      ::close(fd);
      throw;
    }
    ::close(fd);
    return;
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

  const int flags = O_CREAT | O_WRONLY | O_TRUNC;
  bool want_odirect = conn.use_odirect && !odirect_already_refused(conn) &&
                      odirect_alignment_ok(conn.disk_block_size, len, buf);

  bool opened_direct = false;
  int fd = open_maybe_direct(tmp_path.c_str(), flags, 0644, want_odirect,
                             &opened_direct);
  if (fd < 0) {
    throw std::runtime_error("open for write failed: " + tmp_path.string() +
                             ": " + strerror(errno));
  }
  if (want_odirect && !opened_direct) {
    note_odirect_refused(conn);
  }

  try {
    write_all(fd, buf, len);
  } catch (const IoError& e) {
    ::close(fd);
    // O_DIRECT accepted at open() and refused at write() is the live failure
    // on XFS with a host buffer that is length aligned but not address
    // aligned. Without this branch every store fails and the tier stays empty
    // while reporting healthy, so retry buffered once before giving up.
    if (!opened_direct || !is_direct_io_refusal(e.err)) {
      std::filesystem::remove(tmp_path);
      throw;
    }
    note_odirect_refused(conn);
    fd = ::open(tmp_path.c_str(), flags, 0644);
    if (fd < 0) {
      std::filesystem::remove(tmp_path);
      throw std::runtime_error("open for buffered write failed: " +
                               tmp_path.string() + ": " + strerror(errno));
    }
    try {
      write_all(fd, buf, len);
    } catch (...) {
      ::close(fd);
      std::filesystem::remove(tmp_path);
      throw;
    }
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
