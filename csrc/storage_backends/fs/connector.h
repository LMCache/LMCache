// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace lmcache {
namespace connector {

// Key encoding constants — must match fs_l2_adapter.py
static constexpr char KEY_SEP = '@';
static constexpr const char* PATH_SLASH_REPLACEMENT = "-SEP-";
static constexpr const char* FILE_EXT = ".data";
static constexpr const char* TMP_EXT = ".tmp";

// Bytes of reads the connector keeps outstanding against storage when
// read_io_depth is on and no explicit figure is configured.
//
// Throughput is set by bytes in flight, not by object count, so this is
// the figure that decides it.  1536 MiB is a compromise chosen from a
// sweep of 96 MiB to 6 GiB across four conditions, as the value whose
// WORST case is best rather than the value that wins any single one:
//
//   local NVMe array, O_DIRECT     99.0% of the best pinned budget
//   single local NVMe              89.3%
//   8 ms per-read latency          98.9%
//   32 ms per-read latency         99.8%
//
// The asymmetry is what sets it.  At 32 ms, which is the range
// network-attached storage runs at, 768 MiB delivers 58% of what 1536 MiB
// does and 384 MiB delivers 32%.  Being too large is not free but is much
// cheaper: the worst case measured is the single NVMe giving up 10.7%
// between 96 MiB and 1536 MiB.  A deployment that knows its storage can
// do better by pinning its own value, and a single slow device is the
// case most worth pinning.  See
// docs/design/v1/distributed/l2_adapters/native-connector-read-depth.md.
static constexpr size_t kDefaultReadMaxBytesInFlight = size_t{1536} << 20;

// Per-worker connection state for the FS connector.
// Each worker maintains its own I/O buffer for O_DIRECT.
struct WorkerFSConn {
  std::filesystem::path base_path;
  std::filesystem::path tmp_dir;  // empty if not configured
  bool use_odirect = false;
  size_t disk_block_size = 0;
  // If > 0, trigger filesystem readahead by issuing a small
  // initial read of this many bytes before reading the rest.
  size_t read_ahead_size = 0;
};

class FSConnector : public ConnectorBase<WorkerFSConn> {
 public:
  // read_io_depth: reader threads dedicated to executing reads.  Zero
  //   keeps the legacy path, where reads run on the worker threads and
  //   the depth against the device therefore equals num_workers.
  // read_max_bytes_in_flight: bytes this connector may keep outstanding,
  //   shared across its workers.  Zero selects
  //   kDefaultReadMaxBytesInFlight when read_io_depth is positive.
  FSConnector(std::string base_path, int num_workers,
              std::string relative_tmp_dir = "", bool use_odirect = false,
              size_t read_ahead_size = 0, int read_io_depth = 0,
              size_t read_max_bytes_in_flight = 0);
  ~FSConnector() override;

 protected:
  WorkerFSConn create_connection() override;
  void do_single_get(WorkerFSConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerFSConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerFSConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerFSConn& conn, const std::string& key) override;

 private:
  // Build the filesystem-safe filename from a serialized key string.
  //
  // Input key (from NativeConnectorL2Adapter._object_key_to_string):
  //   Unsalted: "{model}@{kv_rank:08x}@{hash.hex()}"
  //   Salted  : "{model}@{kv_rank:08x}@{hash.hex()}@{cache_salt}"
  //
  // Output filename (matching fs_l2_adapter.py._object_key_to_filename):
  //   Unsalted: "{safe_model}@{kv_rank:#010x}@{hash.hex()}.data"
  //   Salted  : "{safe_model}@{kv_rank:#010x}@{hash.hex()}@{cache_salt}.data"
  //
  // Differences from the input: '/' in model becomes '-SEP-', kv_rank
  // gains a '0x' prefix, and '.data' is appended. Both model_name and
  // cache_salt are forbidden from containing '@' (enforced on the
  // Python side), so the parse is unambiguous.
  static std::string key_to_filename(const std::string& key);

  static std::string replace_all(const std::string& str,
                                 const std::string& from,
                                 const std::string& to);

  std::string base_path_;
  std::string relative_tmp_dir_;
  bool use_odirect_;
  size_t disk_block_size_;
  size_t read_ahead_size_;
};

}  // namespace connector
}  // namespace lmcache
