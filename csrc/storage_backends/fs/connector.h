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
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lmcache {
namespace connector {

// Key encoding constants — must match fs_l2_adapter.py
static constexpr const char* TMP_EXT = ".tmp";

// Bytes of reads kept outstanding against the device when num_read_workers is
// on and no explicit figure is configured.  Chosen as the value whose worst
// case across a local array, a single NVMe and 8 / 32 ms per-read latency
// is best; the sweep is in
// docs/design/v1/distributed/l2_adapters/fs_native_read_depth.md.
static constexpr size_t kDefaultReadMaxBytesInFlight = size_t{1536} << 20;
// Reads always admitted whatever the budget, so an object larger than half
// of it still overlaps with the next one instead of draining the device
// between them: one 1 GiB object alone reads ~40 GB/s on an array that
// two in flight bring to ~49.
static constexpr size_t kMinReadsInFlight = 2;

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
  // num_read_workers: reader threads that run lookups and reads, and so the
  //   maximum reads in flight.  Zero keeps the legacy path, where both run
  //   on the worker threads and the read depth against the device equals
  //   num_workers.
  // read_max_bytes_in_flight: bytes kept outstanding, connector-wide.
  //   Zero selects kDefaultReadMaxBytesInFlight when num_read_workers is
  //   positive.
  FSConnector(std::string base_path, int num_workers,
              std::string relative_tmp_dir = "", bool use_odirect = false,
              size_t read_ahead_size = 0, int num_read_workers = 0,
              size_t read_max_bytes_in_flight = 0,
              WorkerPoolConfig worker_pool_config = {});
  ~FSConnector() override;

  // The byte budget in force, or 0 when there is no read pool.
  size_t read_budget_bytes() const;

  // Dispatched directly to the read pool when one is configured, so lookups
  // and reads do not queue behind background writes on the shared workers.
  uint64_t submit_batch_get(const std::vector<std::string>& keys,
                            const std::vector<void*>& bufs,
                            const std::vector<size_t>& lens,
                            size_t batch_chunk_num_bytes) override;
  uint64_t submit_batch_exists(const std::vector<std::string>& keys) override;

 protected:
  WorkerFSConn create_connection() override;
  void do_single_get(WorkerFSConn& conn, const std::string& key, void* buf,
                     size_t len, size_t chunk_size) override;
  void do_single_set(WorkerFSConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;
  bool do_single_exists(WorkerFSConn& conn, const std::string& key) override;
  DeleteResult do_single_delete_result(WorkerFSConn& conn,
                                       const std::string& key) override;

  void on_workers_stopped() override;

 private:
  // A lookup or read batch queued on the pool, shaped like Request: buf_* are
  // empty for lookups.  keys[0..next_idx) are claimed by readers and
  // `remaining` counts keys not yet finished.
  struct PoolBatch {
    uint64_t future_id = 0;
    std::vector<std::string> keys;
    std::vector<void*> buf_ptrs;
    std::vector<size_t> buf_lens;
    size_t batch_chunk_num_bytes = 0;
    std::vector<uint8_t> ok;  // per-key result, moved into the Completion
    size_t next_idx = 0;
    size_t remaining = 0;
    bool any_failed = false;  // set by lookups; reads fail per key instead
    std::string first_error;
  };
  using PoolQueue = std::deque<std::shared_ptr<PoolBatch>>;

  std::shared_ptr<PoolBatch> new_pool_batch(
      const std::vector<std::string>& keys);
  // Queues `batch` unless the pool is stopping, and wakes a reader if the
  // batch can start now.
  void enqueue_pool_batch(PoolQueue& queue, std::shared_ptr<PoolBatch> batch);

  // Whether an object of `len` bytes may be dispatched now.  Requires
  // read_mu_.
  bool read_budget_admits(size_t len) const;
  // Whether a lookup or a budget-admissible read can be popped now.  Requires
  // read_mu_.
  bool has_ready_read_work() const;

  void start_read_pool(int num_read_workers);
  void stop_read_pool();
  void read_thread_main(WorkerFSConn& conn);

  // Claim one key, run it unlocked, and record it.  Called and return with
  // `lk` holding read_mu_.
  void run_lookup(WorkerFSConn& conn, std::unique_lock<std::mutex>& lk);
  void run_read(WorkerFSConn& conn, std::unique_lock<std::mutex>& lk);
  // Claims the next key of the front batch, dequeuing the batch once all its
  // keys are claimed.  Requires read_mu_.
  static std::pair<std::shared_ptr<PoolBatch>, size_t> claim_next_key(
      PoolQueue& queue);
  // Releases `lk` and wakes one more reader if work is ready; each woken
  // reader does the same, so an N-key batch wakes at most N idle readers.
  void unlock_and_wake_next(std::unique_lock<std::mutex>& lk);
  // Records key `idx`; the reader finishing the last key publishes the batch
  // completion, releasing read_mu_ around push_completion() so it is never
  // held together with comp_mu_.
  void finish_key(std::unique_lock<std::mutex>& lk, PoolBatch& batch,
                  size_t idx, uint8_t ok);

  std::string base_path_;
  std::string relative_tmp_dir_;
  bool use_odirect_;
  size_t disk_block_size_;
  size_t read_ahead_size_;

  // Read pool.  read_conns_ is sized once and never resized, so reader
  // references stay valid; read_max_bytes_in_flight_ is 0 without a pool
  // and never 0 with one.  Everything after read_mu_ is guarded by it.
  size_t read_max_bytes_in_flight_;
  std::vector<WorkerFSConn> read_conns_;
  std::vector<std::thread> read_threads_;
  std::mutex read_mu_;
  std::condition_variable read_cv_;  // ready work arrived or stop requested
  // Lookups carry no payload and unblock load planning, so readers drain
  // exists_queue_ ahead of get_queue_ without charging the byte budget.
  PoolQueue exists_queue_;
  PoolQueue get_queue_;
  size_t read_bytes_in_flight_ = 0;  // dispatched and not yet finished
  size_t reads_in_flight_ = 0;       // objects behind read_bytes_in_flight_
  bool read_stop_ = false;
};

}  // namespace connector
}  // namespace lmcache
