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

#include <pybind11/pybind11.h>

namespace py = pybind11;

enum class Op : uint8_t;
struct Request;
struct Completion;

class MultiRESPClient {
 public:
  MultiRESPClient(std::string host, int port, size_t chunk_bytes,
                  int num_workers);
  ~MultiRESPClient();

  MultiRESPClient(const MultiRESPClient&) = delete;
  MultiRESPClient& operator=(const MultiRESPClient&) = delete;

  int event_fd() const;

  uint64_t submit_get(const std::string& key, py::memoryview mv);
  uint64_t submit_set(const std::string& key, py::memoryview mv);
  uint64_t submit_exists(const std::string& key);

  uint64_t submit_batch_get(const std::vector<std::string>& keys,
                            py::list memviews);
  uint64_t submit_batch_set(const std::vector<std::string>& keys,
                            py::list memviews);

  py::list drain_completions();

  void close();

 private:
  void enqueue_request(Request&& req);
  uint64_t submit_with_buffer(Op op, const std::string& key, py::memoryview mv);
  void push_completion(Completion&& c);
  void drain_eventfd_();
  void signal_eventfd_();
  void worker_loop();

  std::string host_;
  int port_;
  size_t chunk_bytes_;
  int num_workers_;

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
