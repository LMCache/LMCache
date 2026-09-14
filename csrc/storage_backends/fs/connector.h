// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace lmcache {
namespace connector {

// Key encoding constants — must match fs_l2_adapter.py
static constexpr char KEY_SEP = '@';
static constexpr const char* PATH_SLASH_REPLACEMENT = "-SEP-";
static constexpr const char* FILE_EXT = ".data";
static constexpr const char* TMP_EXT = ".tmp";

// Bytes of reads kept outstanding against the device when read_io_depth is
// on and no explicit figure is configured.  Chosen as the value whose worst
// case across a local array, a single NVMe and 8 / 32 ms per-read latency
// is best; the sweep is in
// docs/design/v1/distributed/l2_adapters/fs_native_read_depth.md.
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
  // read_io_depth: reader threads, and so the maximum reads in flight.
  //   Zero keeps the legacy path, where reads run on the worker threads
  //   and the depth against the device equals num_workers.
  // read_max_bytes_in_flight: bytes kept outstanding, split into equal
  //   per-worker shares.  Zero selects kDefaultReadMaxBytesInFlight when
  //   read_io_depth is positive.
  // Throws std::runtime_error if read_io_depth is negative; threads
  // started before a failure are joined first.
  FSConnector(std::string base_path, int num_workers,
              std::string relative_tmp_dir = "", bool use_odirect = false,
              size_t read_ahead_size = 0, int read_io_depth = 0,
              size_t read_max_bytes_in_flight = 0);
  ~FSConnector() override;

  // The byte budget in force, or 0 when there is no read pool.
  size_t read_budget_bytes() const;

 protected:
  WorkerFSConn create_connection() override;
  void do_single_get(WorkerFSConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerFSConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerFSConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerFSConn& conn, const std::string& key) override;

  // With a read pool, split a GET batch across workers only as far as
  // leaves every tile read_io_depth objects deep: a worker blocks on its
  // own tile, so tiles shorter than the depth would idle readers.
  size_t choose_num_tiles(Op op, size_t num_items) const override;

  // Through the read pool when one is configured, otherwise the base
  // class's legacy path.  Per-key failures are reported the same way.
  void do_batch_get(WorkerFSConn& conn, const Request& req) override;

  // Stops the pool.  Only safe after the workers are joined: a worker may
  // be waiting on a group only the readers can finish.
  void on_workers_stopped() override;

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

  // Completion state of one group, on the waiting worker's stack.  Guarded
  // by read_mu_, which also keeps it alive: the waiter destroys it only
  // after taking that lock and seeing remaining == 0.
  struct ReadGroup {
    std::vector<uint8_t> ok;  // one flag per object of the group
    size_t remaining = 0;
  };

  // One object to read; `req` outlives it because its worker waits.
  struct ReadTask {
    const Request* req = nullptr;
    size_t index = 0;  // index into req->keys
    size_t local = 0;  // index into group->ok
    ReadGroup* group = nullptr;
  };

  // Hand a tile to the readers a group at a time, waiting for each.
  void do_batch_get_pooled(const Request& req);

  // Per-worker share of the byte budget.
  size_t worker_read_budget_bytes() const;

  void start_read_pool(int read_io_depth);
  void stop_read_pool();
  void read_thread_main(WorkerFSConn& conn);

  std::string base_path_;
  std::string relative_tmp_dir_;
  bool use_odirect_;
  size_t disk_block_size_;
  size_t read_ahead_size_;

  // Read pool.  read_conns_ is sized once and never resized, so reader
  // references stay valid; read_max_bytes_in_flight_ is 0 without a pool
  // and never 0 with one.
  size_t read_max_bytes_in_flight_;
  std::vector<WorkerFSConn> read_conns_;
  std::vector<std::thread> read_threads_;
  std::mutex read_mu_;
  std::condition_variable read_cv_;       // a task is available
  std::condition_variable read_done_cv_;  // some group finished
  std::list<ReadTask> read_queue_;
  bool read_stop_ = false;  // guarded by read_mu_
};

}  // namespace connector
}  // namespace lmcache
