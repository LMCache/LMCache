// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

enum class Op : uint8_t { BATCH_TILE_GET, BATCH_TILE_SET, BATCH_TILE_EXISTS };

/* shared communication state between threads executing a single batch operation
all threads need to complete before the completion is sent */
/* tiling refers to dividing work for batched operations between threads
beforehand */
struct BatchState {
  // the number of tiles left to finish
  // we can only send a completion when the last tile is finished
  std::atomic<uint32_t> remaining_tiles{0};
  std::atomic<bool> any_failed{false};

  std::mutex err_mu;
  std::string first_error;

  // for batch exists, store the boolean results (0/1)
  // IMPORTANT: not vector<bool> due to concurrent write data race
  std::vector<uint8_t> exists_results;

  // track batch operation type to avoid fragile req.op checks
  Op batch_op;
};

/*
LIFETIME GUARANTEE:
We have a strict assumption that Python will NOT clean up any buffer memory
before all C++ operations finish. This is guaranteed by the Python-side design
where the caller holds references to all buffers until drain_completions()
returns the corresponding future_id. Therefore, we do NOT need to track
buf_owner references or acquire the GIL to prevent premature cleanup.
We can safely use raw pointers extracted under the GIL without additional
lifetime management on the C++ side.
*/

struct Request {
  // the completion also has a future_id
  // the caller is responsible for matching the request to the completion
  uint64_t future_id = 0;
  Op op;

  // all operations use the batched structure (even single-item operations
  // are treated as batches of size 1)
  std::vector<std::string> keys;
  std::vector<void*> buf_ptrs;
  std::vector<size_t> buf_lens;

  // shared batch state between threads executing a single batch operation
  // so that they can coordinate when to send the completion
  std::shared_ptr<BatchState> batch;

  // for batch exists tiles, track which indices this tile is responsible for
  size_t start_idx = 0;

  // batch_chunk_num_bytes for GET/SET operations (passed per-operation, not
  // per-connection)
  size_t batch_chunk_num_bytes = 0;
};

struct Completion {
  // the request also has a future_id
  // the caller is responsible for matching the completion to the request
  uint64_t future_id = 0;

  // did the operation succeed?
  bool ok = true;

  // for EXISTS operations (both single and batch), store boolean results as
  // bytes (0/1) single EXISTS will have 1 element, batch EXISTS will have N
  // elements (no result in the completion for SET and GET)
  std::vector<uint8_t> result_bytes;

  // error string if operation failed
  std::string error;
};

class MultiRESPClient {
 public:
  MultiRESPClient(std::string host, int port, int num_workers,
                  std::string username = "", std::string password = "");
  ~MultiRESPClient();

  MultiRESPClient(const MultiRESPClient&) = delete;
  MultiRESPClient& operator=(const MultiRESPClient&) = delete;

  int event_fd() const;

  uint64_t submit_batch_get(const std::vector<std::string>& keys,
                            const std::vector<void*>& bufs,
                            const std::vector<size_t>& lens,
                            size_t batch_chunk_num_bytes);
  uint64_t submit_batch_set(const std::vector<std::string>& keys,
                            const std::vector<void*>& bufs,
                            const std::vector<size_t>& lens,
                            size_t batch_chunk_num_bytes);
  uint64_t submit_batch_exists(const std::vector<std::string>& keys);

  std::vector<Completion> drain_completions();

  void close();

 private:
  void enqueue_request(Request&& req);
  void push_completion(Completion&& c);
  void drain_eventfd_();
  void signal_eventfd_();
  void worker_loop();

  std::string host_;
  int port_;
  int num_workers_;
  std::string username_;
  std::string password_;

  int efd_ = -1;

  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::atomic<uint64_t> next_future_id_{1};

  // we treat eventfd not as a counter, but as a binary wakeup flag.
  // true: Python has been signaled (or will be)
  // false: Python is asleep, no wakeup pending
  std::atomic<bool> signaled_{false};

  /*
  SQ/CQ Design
  */
  std::mutex req_mu_;
  std::condition_variable req_cv_;
  // SUBMISSION QUEUE
  std::queue<Request> requests_;

  std::mutex comp_mu_;
  // COMPLETION QUEUE
  std::queue<Completion> completions_;

  std::vector<std::thread> workers_;

  // track worker socket fds so we can shutdown during close()
  std::mutex worker_fds_mu_;
  std::vector<int> worker_fds_;
};
