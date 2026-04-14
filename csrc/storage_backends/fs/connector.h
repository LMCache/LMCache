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
  FSConnector(std::string base_path, int num_workers,
              std::string relative_tmp_dir = "", bool use_odirect = false,
              size_t read_ahead_size = 0);
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
  // The key string has the format: model_name@kv_rank_hex@chunk_hash_hex
  // Output filename: model_name_safe@0xkv_rank_hex@chunk_hash_hex.data
  //
  // NOTE: the key string from NativeConnectorL2Adapter uses
  //   _object_key_to_string: "{model}@{kv_rank:08x}@{hash.hex()}"
  // while fs_l2_adapter.py uses _object_key_to_filename:
  //   "{safe_model}@{kv_rank:#010x}@{hash.hex()}.data"
  //
  // The difference is the 0x prefix in kv_rank. We handle
  // this by re-encoding here to match the Python FS layout.
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
