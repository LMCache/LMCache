// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "connector_interface.h"
#include "connector_types.h"
#include <sys/eventfd.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace lmcache {
namespace connector {

/*
this base needs to have at least four methods be overridden by the derived
class:
- 1. create_connection() e.g. construct TCP socket or RDMA registration
- 2. do_single_get()
- 3. do_single_set()
- 4. do_single_exists()

see the RedisConnector (csrc/redis/) implementing the RESP2 protocol over TCP
for an example
*/
template <typename ConnectionType>
class ConnectorBase : public IStorageConnector {
 public:
  ConnectorBase(int num_workers) : num_workers_(num_workers) {
    if (num_workers_ <= 0) {
      throw std::runtime_error("num_workers must be > 0");
    }

    // create eventfd for async notification
    // EFD_NONBLOCK: read() and write() are non-blocking
    // EFD_CLOEXEC: close on exec
    efd_ = ::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    if (efd_ < 0) {
      throw std::runtime_error("failed to create eventfd");
    }
  }

  virtual ~ConnectorBase() { close(); }

  ConnectorBase(const ConnectorBase&) = delete;
  ConnectorBase& operator=(const ConnectorBase&) = delete;

  int event_fd() const override { return efd_; }

  uint64_t submit_batch_get(const std::vector<std::string>& keys,
                            const std::vector<void*>& bufs,
                            const std::vector<size_t>& lens,
                            size_t batch_chunk_num_bytes) override {
    validate_batch_inputs(keys, bufs, lens);

    size_t num_items = keys.size();
    auto [batch_future_id, batch_state, num_tiles, tile_size] =
        prepare_batch_operation(num_items, Op::BATCH_TILE_GET);

    // fan out work to threads
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      auto tile_req = create_tile_request(
          keys, bufs, lens, tile_idx, tile_size, num_items, batch_future_id,
          batch_state, Op::BATCH_TILE_GET, batch_chunk_num_bytes);
      enqueue_request(std::move(tile_req));
    }

    return batch_future_id;
  }

  uint64_t submit_batch_set(const std::vector<std::string>& keys,
                            const std::vector<void*>& bufs,
                            const std::vector<size_t>& lens,
                            size_t batch_chunk_num_bytes) override {
    validate_batch_inputs(keys, bufs, lens);

    size_t num_items = keys.size();
    auto [batch_future_id, batch_state, num_tiles, tile_size] =
        prepare_batch_operation(num_items, Op::BATCH_TILE_SET);

    // fan out work to threads
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      auto tile_req = create_tile_request(
          keys, bufs, lens, tile_idx, tile_size, num_items, batch_future_id,
          batch_state, Op::BATCH_TILE_SET, batch_chunk_num_bytes);
      enqueue_request(std::move(tile_req));
    }

    return batch_future_id;
  }

  uint64_t submit_batch_exists(const std::vector<std::string>& keys) override {
    if (keys.empty()) {
      throw std::runtime_error("keys list is empty");
    }

    size_t num_items = keys.size();
    auto [batch_future_id, batch_state, num_tiles, tile_size] =
        prepare_batch_operation(num_items, Op::BATCH_TILE_EXISTS);

    // pre-allocate results vector with correct size
    batch_state->exists_results.assign(num_items, 0);

    // fan out work to threads
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      size_t start = tile_idx * tile_size;
      size_t end = std::min(start + tile_size, num_items);

      Request tile_req;
      tile_req.op = Op::BATCH_TILE_EXISTS;
      tile_req.future_id = batch_future_id;
      tile_req.batch = batch_state;
      tile_req.start_idx = start;

      for (size_t i = start; i < end; ++i) {
        tile_req.keys.push_back(keys[i]);
      }

      enqueue_request(std::move(tile_req));
    }

    return batch_future_id;
  }

  std::vector<Completion> drain_completions() override {
    // Drain the eventfd that triggered this drain_completions callback
    drain_eventfd_();

    std::vector<Completion> completions_list;

    for (;;) {
      Completion c;
      {
        std::lock_guard<std::mutex> lk(comp_mu_);
        if (completions_.empty()) {
          signaled_.store(false, std::memory_order_release);
          if (!completions_.empty() &&
              !signaled_.exchange(true, std::memory_order_acq_rel)) {
            uint64_t x = 1;
            ::write(efd_, &x, sizeof(x));
          }
          break;
        }

        c = std::move(completions_.front());
        completions_.pop();
      }
      completions_list.push_back(std::move(c));
    }

    return completions_list;
  }

  void close() override {
    if (closed_.exchange(true, std::memory_order_acq_rel)) {
      return;  // Already closed
    }

    // Signal all worker threads to stop
    stop_.store(true, std::memory_order_release);
    req_cv_.notify_all();

    // Shutdown all connections (derived class specific)
    shutdown_connections();

    // Join all worker threads
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }

    // Close eventfd
    if (efd_ >= 0) {
      ::close(efd_);
      efd_ = -1;
    }

    // Clear queues (no GIL needed - python guarantees buffers stay alive)
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      while (!requests_.empty()) {
        requests_.pop();
      }
    }
    {
      std::lock_guard<std::mutex> lk(comp_mu_);
      while (!completions_.empty()) {
        completions_.pop();
      }
    }
  }

 protected:
  // call this at the END of your derived class constructor
  void start_workers() {
    workers_.reserve(static_cast<size_t>(num_workers_));
    for (int i = 0; i < num_workers_; i++) {
      workers_.emplace_back([this]() { this->worker_loop(); });
    }
  }

  virtual ConnectionType create_connection() = 0;
  virtual void do_single_get(ConnectionType& conn, const std::string& key,
                             void* buf, size_t len, size_t chunk_size) = 0;
  virtual void do_single_set(ConnectionType& conn, const std::string& key,
                             const void* buf, size_t len,
                             size_t chunk_size) = 0;
  virtual bool do_single_exists(ConnectionType& conn,
                                const std::string& key) = 0;
  virtual void shutdown_connections() {}

  bool is_stopping() const { return stop_.load(std::memory_order_acquire); }

 private:
  void validate_batch_inputs(const std::vector<std::string>& keys,
                             const std::vector<void*>& bufs,
                             const std::vector<size_t>& lens) {
    if (keys.size() != bufs.size() || keys.size() != lens.size()) {
      throw std::runtime_error("keys, bufs, and lens size mismatch");
    }
    if (keys.empty()) {
      throw std::runtime_error("keys list is empty");
    }
  }

  // returns: (batch_future_id, batch_state, num_tiles, tile_size)
  std::tuple<uint64_t, std::shared_ptr<BatchState>, size_t, size_t>
  prepare_batch_operation(size_t num_items, Op op) {
    // divide work evenly between workers into tiles
    size_t num_tiles =
        std::min<size_t>(num_workers_, num_items);  // avoid empty tiles
    size_t tile_size = (num_items + num_tiles - 1) / num_tiles;  // round up

    // create shared batch state
    uint64_t batch_future_id =
        next_future_id_.fetch_add(1, std::memory_order_relaxed);
    auto batch_state = std::make_shared<BatchState>();
    batch_state->remaining_tiles.store(num_tiles, std::memory_order_relaxed);
    batch_state->batch_op = op;

    return {batch_future_id, batch_state, num_tiles, tile_size};
  }

  Request create_tile_request(const std::vector<std::string>& keys,
                              const std::vector<void*>& bufs,
                              const std::vector<size_t>& lens, size_t tile_idx,
                              size_t tile_size, size_t num_items,
                              uint64_t batch_future_id,
                              std::shared_ptr<BatchState> batch_state, Op op,
                              size_t batch_chunk_num_bytes) {
    size_t start = tile_idx * tile_size;
    size_t end = std::min(start + tile_size, num_items);  // clip last tile

    Request tile_req;
    tile_req.op = op;
    tile_req.future_id = batch_future_id;
    tile_req.batch = batch_state;
    tile_req.batch_chunk_num_bytes = batch_chunk_num_bytes;

    for (size_t i = start; i < end; ++i) {
      tile_req.keys.push_back(keys[i]);
      tile_req.buf_ptrs.push_back(bufs[i]);
      tile_req.buf_lens.push_back(lens[i]);
    }

    return tile_req;
  }

  void enqueue_request(Request&& req) {
    {
      std::lock_guard<std::mutex> lk(req_mu_);
      requests_.push(std::move(req));
    }
    req_cv_.notify_one();
  }

  void push_completion(Completion&& c) {
    {
      std::lock_guard<std::mutex> lk(comp_mu_);
      completions_.push(std::move(c));
    }
    signal_eventfd_();
  }

  void drain_eventfd_() {
    // loop to consume all writes that happened since last drain
    for (;;) {
      uint64_t x;
      ssize_t r = ::read(efd_, &x, sizeof(x));
      if (r == static_cast<ssize_t>(sizeof(x))) {
        continue;  // keep draining
      }
      if (r < 0) {
        if (errno == EINTR) {
          continue;  // retry on EINTR
        }
        if (errno == EAGAIN) {
          break;  // drained (no more data)
        }
      }
      break;
    }
  }

  void signal_eventfd_() {
    bool already_signaled = signaled_.exchange(true, std::memory_order_acq_rel);
    if (already_signaled) return;  // only one signal at a time

    uint64_t x = 1;
    for (;;) {
      ssize_t w = ::write(efd_, &x, sizeof(x));
      if (w == static_cast<ssize_t>(sizeof(x))) {
        return;  // success
      }
      if (w < 0) {
        if (errno == EINTR) {
          continue;  // retry on EINTR
        }
        throw std::runtime_error("eventfd write failed unexpectedly");
      }
      throw std::runtime_error("partial write to eventfd");
    }
  }

  void worker_loop() {
    try {
      // create connection (derived class specific)
      ConnectionType conn = create_connection();

      for (;;) {
        Request req;

        // 1. grab a request from the submission queue
        {
          std::unique_lock<std::mutex> lk(req_mu_);
          req_cv_.wait(lk, [&] {
            return stop_.load(std::memory_order_acquire) || !requests_.empty();
          });
          if (stop_.load(std::memory_order_acquire) && requests_.empty()) {
            break;  // exit loop
          }
          req = std::move(requests_.front());
          requests_.pop();
        }

        Completion comp;
        comp.future_id = req.future_id;

        // 2. execute the requested operation
        try {
          switch (req.op) {
            case Op::BATCH_TILE_GET:
              for (size_t i = 0; i < req.keys.size(); ++i) {
                do_single_get(conn, req.keys[i], req.buf_ptrs[i],
                              req.buf_lens[i], req.batch_chunk_num_bytes);
              }
              comp.ok = true;
              break;

            case Op::BATCH_TILE_SET:
              for (size_t i = 0; i < req.keys.size(); ++i) {
                do_single_set(conn, req.keys[i], req.buf_ptrs[i],
                              req.buf_lens[i], req.batch_chunk_num_bytes);
              }
              comp.ok = true;
              break;

            case Op::BATCH_TILE_EXISTS:
              for (size_t i = 0; i < req.keys.size(); ++i) {
                bool exists = do_single_exists(conn, req.keys[i]);
                // Write result as uint8_t (0/1) to avoid vector<bool> data race
                req.batch->exists_results[req.start_idx + i] = exists ? 1 : 0;
              }
              comp.ok = true;
              break;
          }
        } catch (const std::exception& e) {
          comp.ok = false;
          comp.error = e.what();
          // if shutting down, errors are expected
          if (stop_.load(std::memory_order_acquire)) {
            break;  // exit without pushing completion
          }
        }

        // 3. update shared batch state and push completion when done
        handle_tile_completion(req, comp);
      }
    } catch (const std::exception& e) {
      fprintf(stderr, "[LMCache Connector Worker Error] %s\n", e.what());
    } catch (...) {
      fprintf(stderr, "[LMCache Connector Worker Error] Unknown exception\n");
    }
  }

  void handle_tile_completion(const Request& req, const Completion& comp) {
    // record failure if any
    if (!comp.ok) {
      req.batch->any_failed.store(true, std::memory_order_relaxed);
      std::lock_guard<std::mutex> lk(req.batch->err_mu);
      if (req.batch->first_error.empty()) {
        req.batch->first_error = comp.error;
      }
    }

    // check if this is the last tile to complete
    uint32_t tiles_left =
        req.batch->remaining_tiles.fetch_sub(1, std::memory_order_relaxed) - 1;

    if (tiles_left == 0) {
      // last tile to finish - emit single completion for entire batch
      Completion batch_comp;
      batch_comp.future_id = req.future_id;
      batch_comp.ok = !req.batch->any_failed.load(std::memory_order_relaxed);
      if (!batch_comp.ok) {
        std::lock_guard<std::mutex> lk(req.batch->err_mu);
        batch_comp.error = req.batch->first_error;
      }
      // for batch exists, move the results
      if (req.batch->batch_op == Op::BATCH_TILE_EXISTS) {
        batch_comp.result_bytes = std::move(req.batch->exists_results);
      }
      push_completion(std::move(batch_comp));
    }
  }

 protected:
  int num_workers_;

  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::atomic<uint64_t> next_future_id_{1};

 private:
  int efd_ = -1;

  // treat eventfd as a binary wakeup flag:
  // true: Python has been signaled (or will be)
  // false: Python is asleep, no wakeup pending
  std::atomic<bool> signaled_{false};

  // submission queue (SQ)
  std::mutex req_mu_;
  std::condition_variable req_cv_;
  std::queue<Request> requests_;

  // completion queue (CQ)
  std::mutex comp_mu_;
  std::queue<Completion> completions_;

  std::vector<std::thread> workers_;
};

}  // namespace connector
}  // namespace lmcache
