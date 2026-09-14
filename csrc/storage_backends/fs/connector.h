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

// Bytes of reads the connector keeps outstanding against the device when
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
  // read_io_depth: reader threads dedicated to executing reads, and so
  //   the maximum reads in flight.  Zero keeps the legacy path, where
  //   reads run on the worker threads and the depth against the device
  //   therefore equals num_workers.  Each reader thread holds one open
  //   file at a time, so this is also the ceiling on those.
  // read_max_bytes_in_flight: bytes this connector may keep outstanding,
  //   split into equal shares, one per worker.  Zero selects
  //   kDefaultReadMaxBytesInFlight when read_io_depth is positive.
  // Throws std::runtime_error if read_io_depth is negative.  If starting
  // the pool or the workers fails, every thread already started is joined
  // before the exception propagates.
  FSConnector(std::string base_path, int num_workers,
              std::string relative_tmp_dir = "", bool use_odirect = false,
              size_t read_ahead_size = 0, int read_io_depth = 0,
              size_t read_max_bytes_in_flight = 0);
  ~FSConnector() override;

  // Bytes this connector allows outstanding against the device.
  //
  // Returns the byte budget in force, which is the configured value or
  // kDefaultReadMaxBytesInFlight when none was given, or 0 when there is
  // no read pool and so nothing to bound.  Exposed so a caller can report
  // it alongside its other storage statistics.
  size_t read_budget_bytes() const;

 protected:
  WorkerFSConn create_connection() override;
  void do_single_get(WorkerFSConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerFSConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerFSConn& conn, const std::string& key) override;
  bool do_single_delete(WorkerFSConn& conn, const std::string& key) override;

  // With a read pool, a GET batch is split across workers only as far as
  // leaves every tile at least read_io_depth objects deep; a smaller batch
  // stays one tile.  The base class's split into num_workers tiles would
  // cap reads in flight at num_workers however large read_io_depth is:
  // each worker only ever hands its OWN tile to the pool and then blocks
  // on it, so the depth would be inert for any batch smaller than
  // num_workers x depth, which is most real requests.
  size_t choose_num_tiles(Op op, size_t num_items) const override;

  // Read one tile of a batch: through the read pool when one is
  // configured, otherwise the base class's legacy path, one blocking
  // read at a time.  Per-key error
  // tolerance is the same either way: a key whose read fails is marked 0
  // in per_key_results and the rest of the tile still completes.
  void do_batch_get(WorkerFSConn& conn, const Request& req) override;

  // The pool may only stop once the workers have been joined: a worker may
  // be blocked waiting for a group, and only the reader threads can finish
  // it.  The base class calls this from close() at exactly that point.
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

  // Completion state for the objects of one group, living on the stack of
  // the worker thread that is waiting for them.  Both fields are guarded
  // by read_mu_, which is also what keeps the group alive long enough:
  // the waiter can only return, and so destroy it, while holding that
  // lock, and a reader only touches it while holding the same lock.
  struct ReadGroup {
    std::vector<uint8_t> ok;  // one flag per object of the group
    size_t remaining = 0;
  };

  // One object to read.  `req` outlives the task: the worker that owns
  // the request blocks until every task referring to it has finished.
  struct ReadTask {
    const Request* req = nullptr;
    size_t index = 0;  // index into req->keys
    size_t local = 0;  // index into group->ok
    ReadGroup* group = nullptr;
  };

  // Hand a tile's objects to the reader threads, a group at a time, and
  // wait for each group.  Reads run on the pool's own connections, so the
  // calling worker's is idle for the duration.
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

  // Read pool.  read_conns_ is sized once in start_read_pool() and never
  // resized, so each reader thread's reference stays valid; read_threads_
  // being empty is the test for "no pool".  read_max_bytes_in_flight_ is
  // normalised in the constructor: 0 without a pool, never 0 with one.
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
