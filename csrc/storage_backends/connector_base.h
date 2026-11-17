// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "connector_interface.h"
#include "connector_types.h"
#include "event_notifier.h"
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace lmcache {
namespace connector {

struct WorkerPoolConfig {
  // Maps lane key (e.g. "lookup", "retrieve", "store") to worker count.
  // Lane keys not present in this map use the shared num_workers_ pool.
  std::unordered_map<std::string, int> per_op_workers;
};

/*
this base needs to have at least four methods be overridden by the derived
class:
- 1. create_connection() e.g. construct TCP socket or RDMA registration
- 2. do_single_get()
- 3. do_single_set()
- 4. do_single_exists()

optionally override do_single_delete() to support eviction (default returns
false for all keys).

see the RedisConnector (csrc/redis/) implementing the RESP2 protocol over TCP
for an example

SCHEDULING (reads are prioritized over writes):
every submission queue (the shared pool's and each dedicated lane's) has two
classes. GET and EXISTS tiles ("reads", issued by the latency-sensitive
prefetch path) are high priority; SET and DELETE tiles ("writes", issued by
the background store / eviction paths) are low priority.
- a worker always takes a pending read before a pending write
- the default do_batch_set() checks for pending reads after every key and,
  if there is one, stops; the worker puts the unfinished remainder back at
  the front of the write class and serves the read. A read therefore waits
  for at most one in-flight do_single_set() per worker instead of a whole
  tile. Connectors that override do_batch_set() keep their tile atomic
  unless they call yield_to_pending_read() themselves
- to keep writes from starving, after kMaxReadStreak consecutive read tiles
  were taken while a write was waiting, one write tile is taken. Every write
  pickup stores at least one key before it may yield, so writes always make
  progress
A lane dedicated to a single op (see WorkerPoolConfig) only ever holds one
class, so this has no effect there: such ops are already isolated.
*/
template <typename ConnectionType>
class ConnectorBase : public IStorageConnector {
 public:
  ConnectorBase(int num_workers)
      : ConnectorBase(num_workers, WorkerPoolConfig{}) {}

  ConnectorBase(int num_workers, WorkerPoolConfig worker_pool_config)
      : num_workers_(num_workers), worker_pool_config_(worker_pool_config) {
    if (num_workers_ <= 0) {
      throw std::runtime_error("num_workers must be > 0");
    }
    for (auto& [key, count] : worker_pool_config_.per_op_workers) {
      if (count <= 0) {
        throw std::runtime_error("per_op_workers count for '" + key +
                                 "' must be positive");
      }
      if (lane_registry().find(key) == lane_registry().end()) {
        throw std::runtime_error("unknown lane key in per_op_workers: '" + key +
                                 "'");
      }
    }

    // Cross-platform poll-able wakeup fd (eventfd on Linux,
    // self-pipe on macOS). See event_notifier.h.
    notifier_ = make_event_notifier();
  }

  virtual ~ConnectorBase() { close(); }

  ConnectorBase(const ConnectorBase&) = delete;
  ConnectorBase& operator=(const ConnectorBase&) = delete;

  int event_fd() const override { return notifier_->fileno(); }

  uint64_t submit_batch_get(const std::vector<std::string>& keys,
                            const std::vector<void*>& bufs,
                            const std::vector<size_t>& lens,
                            size_t batch_chunk_num_bytes) override {
    validate_batch_inputs(keys, bufs, lens);

    size_t num_items = keys.size();
    auto [batch_future_id, batch_state, num_tiles] =
        prepare_batch_operation(num_items, Op::BATCH_TILE_GET);

    // pre-allocate per-key results for load error tolerance
    batch_state->per_key_results.assign(num_items, 0);

    // fan out work to threads
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      auto tile_req = create_tile_request(
          keys, bufs, lens, tile_idx, num_tiles, num_items, batch_future_id,
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
    auto [batch_future_id, batch_state, num_tiles] =
        prepare_batch_operation(num_items, Op::BATCH_TILE_SET);

    // fan out work to threads
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      auto tile_req = create_tile_request(
          keys, bufs, lens, tile_idx, num_tiles, num_items, batch_future_id,
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
    auto [batch_future_id, batch_state, num_tiles] =
        prepare_batch_operation(num_items, Op::BATCH_TILE_EXISTS);

    // pre-allocate results vector with correct size
    batch_state->per_key_results.assign(num_items, 0);

    // fan out work to threads
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      auto [start, end] = get_tile_range(tile_idx, num_items, num_tiles);

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

  uint64_t submit_batch_delete(const std::vector<std::string>& keys) override {
    if (keys.empty()) {
      throw std::runtime_error("keys list is empty");
    }

    size_t num_items = keys.size();
    auto [batch_future_id, batch_state, num_tiles] =
        prepare_batch_operation(num_items, Op::BATCH_TILE_DELETE);

    // pre-allocate per-key results (1 = deleted, 0 = not found)
    batch_state->per_key_results.assign(num_items, 0);

    // fan out work to threads
    for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
      auto [start, end] = get_tile_range(tile_idx, num_items, num_tiles);

      Request tile_req;
      tile_req.op = Op::BATCH_TILE_DELETE;
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
            notifier_->notify();
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
    shared_queue_.cv.notify_all();
    for (auto& [key, lane] : lanes_) {
      lane->queue.cv.notify_all();
    }

    // Shutdown all connections (derived class specific)
    shutdown_connections();

    // Join all worker threads
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
    for (auto& [key, lane] : lanes_) {
      join_lane_workers(*lane);
    }

    // Derived cleanup that must only run after workers have stopped.
    on_workers_stopped();

    // Close wakeup fd
    if (notifier_) {
      notifier_->close();
    }

    // Clear queues (no GIL needed - python guarantees buffers stay alive)
    clear_queue(shared_queue_);
    for (auto& [key, lane] : lanes_) {
      clear_queue(lane->queue);
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
    // Start dedicated per-op lanes for configured keys.
    for (auto& [key, count] : worker_pool_config_.per_op_workers) {
      auto& lane_ptr = lanes_[key];
      if (!lane_ptr) {
        lane_ptr = std::make_unique<WorkerLane>();
      }
      start_lane_workers(*lane_ptr, count);
    }

    // Start shared pool if any lane key in the registry is not configured.
    bool need_shared = false;
    for (auto& [key, ops] : lane_registry()) {
      if (worker_pool_config_.per_op_workers.find(key) ==
          worker_pool_config_.per_op_workers.end()) {
        need_shared = true;
        break;
      }
    }

    if (need_shared) {
      workers_.reserve(static_cast<size_t>(num_workers_));
      for (int i = 0; i < num_workers_; i++) {
        workers_.emplace_back([this]() { this->worker_loop(); });
      }
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
  virtual bool do_single_delete(ConnectionType& conn, const std::string& key) {
    (void)conn;
    (void)key;
    return false;  // no-op default for backward compat with plugins
  }
  virtual size_t choose_num_tiles(Op op, size_t num_items) const {
    return std::min<size_t>(worker_count_for_op(op), num_items);
  }
  virtual void do_batch_get(ConnectionType& conn, const Request& req) {
    for (size_t i = 0; i < req.keys.size(); ++i) {
      try {
        do_single_get(conn, req.keys[i], req.buf_ptrs[i], req.buf_lens[i],
                      req.batch_chunk_num_bytes);
        req.batch->per_key_results[req.start_idx + i] = 1;
      } catch (const std::exception& e) {
        req.batch->per_key_results[req.start_idx + i] = 0;
        fprintf(stderr, "[LMCache GET] key %s failed: %s\n",
                req.keys[i].c_str(), e.what());
      }
    }
  }
  // Stores the keys of a SET tile, starting at req.resume_idx.
  //
  // The default implementation yields to pending reads between keys (see
  // the SCHEDULING note at the top of this file). An override that never
  // calls yield_to_pending_read() always runs on a whole tile, with
  // req.resume_idx == 0.
  virtual void do_batch_set(ConnectionType& conn, const Request& req) {
    for (size_t i = req.resume_idx; i < req.keys.size(); ++i) {
      do_single_set(conn, req.keys[i], req.buf_ptrs[i], req.buf_lens[i],
                    req.batch_chunk_num_bytes);
      if (yield_to_pending_read(req, i + 1)) {
        return;
      }
    }
  }
  virtual void do_batch_exists(ConnectionType& conn, const Request& req) {
    for (size_t i = 0; i < req.keys.size(); ++i) {
      bool exists = do_single_exists(conn, req.keys[i]);
      req.batch->per_key_results[req.start_idx + i] = exists ? 1 : 0;
    }
  }
  virtual void do_batch_delete(ConnectionType& conn, const Request& req) {
    for (size_t i = 0; i < req.keys.size(); ++i) {
      try {
        bool deleted = do_single_delete(conn, req.keys[i]);
        req.batch->per_key_results[req.start_idx + i] = deleted ? 1 : 0;
      } catch (const std::exception& e) {
        req.batch->per_key_results[req.start_idx + i] = 0;
        fprintf(stderr, "[LMCache DELETE] key %s failed: %s\n",
                req.keys[i].c_str(), e.what());
      }
    }
  }
  virtual void shutdown_connections() {}
  virtual void on_workers_stopped() {}

  bool is_stopping() const { return stop_.load(std::memory_order_acquire); }

  // For do_batch_set() implementations, called from a worker thread after
  // the keys before `next_idx` of the tile `req` were stored successfully.
  //
  // Returns true when a read is waiting for this worker and the tile still
  // has keys left. The caller must then return from do_batch_set() without
  // touching the remaining keys: the worker re-queues the tile and
  // do_batch_set() is called again later with req.resume_idx == next_idx.
  // Returns false when the caller should keep going.
  bool yield_to_pending_read(const Request& req, size_t next_idx) {
    WorkerContext* ctx = worker_context_;
    if (ctx == nullptr || next_idx >= req.keys.size() ||
        ctx->queue->pending_reads.load(std::memory_order_acquire) == 0) {
      return false;
    }
    ctx->yield_idx = next_idx;
    return true;
  }

  int worker_count_for_op(Op op) const {
    const std::string& key = lane_key_for_op(op);
    auto it = worker_pool_config_.per_op_workers.find(key);
    if (it != worker_pool_config_.per_op_workers.end()) {
      return it->second;
    }
    return num_workers_;
  }

 private:
  // A submission queue with two priority classes. reads, writes and
  // read_streak are guarded by mu.
  struct RequestQueue {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<Request> reads;   // GET / EXISTS tiles: high priority
    std::deque<Request> writes;  // SET / DELETE tiles: low priority
    // consecutive read tiles taken while a write tile was waiting
    size_t read_streak = 0;
    // mirror of reads.size(), written under mu and read lock-free by a
    // worker deciding whether its write tile should yield
    std::atomic<size_t> pending_reads{0};
  };

  struct WorkerLane {
    RequestQueue queue;
    std::vector<std::thread> workers;
  };

  // Per-worker-thread state, reachable from do_batch_set() through
  // yield_to_pending_read().
  struct WorkerContext {
    RequestQueue* queue = nullptr;
    // set by yield_to_pending_read(): first unprocessed key of the tile
    // being executed, or kNotYielded
    size_t yield_idx = kNotYielded;
  };

  static constexpr size_t kNotYielded = static_cast<size_t>(-1);

  // max number of consecutive read tiles taken while a write tile is
  // waiting, before one write tile is taken (write starvation guard)
  static constexpr size_t kMaxReadStreak = 32;

  // reads (GET / EXISTS) are served before writes (SET / DELETE)
  static bool is_read_op(Op op) {
    return op == Op::BATCH_TILE_GET || op == Op::BATCH_TILE_EXISTS;
  }

  struct TileRange {
    size_t start;
    size_t end;
  };

  static TileRange get_tile_range(size_t tile_idx, size_t num_items,
                                  size_t num_tiles) {
    // Distribute the remainder across the first tiles so tile sizes differ
    // by at most one. The quotient/remainder calculation also avoids
    // overflow-prone round-up and tile_idx * num_items formulas.
    size_t base_tile_size = num_items / num_tiles;
    size_t larger_tiles = num_items % num_tiles;
    size_t start = tile_idx * base_tile_size + std::min(tile_idx, larger_tiles);
    size_t tile_size = base_tile_size + (tile_idx < larger_tiles ? 1 : 0);
    return {start, start + tile_size};
  }

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

  // returns: (batch_future_id, batch_state, num_tiles)
  std::tuple<uint64_t, std::shared_ptr<BatchState>, size_t>
  prepare_batch_operation(size_t num_items, Op op) {
    size_t num_tiles = choose_num_tiles(op, num_items);
    if (num_tiles == 0 || num_tiles > num_items) {
      throw std::runtime_error(
          "choose_num_tiles must return a value in [1, num_items]");
    }

    // create shared batch state
    uint64_t batch_future_id =
        next_future_id_.fetch_add(1, std::memory_order_relaxed);
    auto batch_state = std::make_shared<BatchState>();
    batch_state->remaining_tiles.store(num_tiles, std::memory_order_relaxed);
    batch_state->batch_op = op;

    return {batch_future_id, batch_state, num_tiles};
  }

  Request create_tile_request(const std::vector<std::string>& keys,
                              const std::vector<void*>& bufs,
                              const std::vector<size_t>& lens, size_t tile_idx,
                              size_t num_tiles, size_t num_items,
                              uint64_t batch_future_id,
                              std::shared_ptr<BatchState> batch_state, Op op,
                              size_t batch_chunk_num_bytes) {
    auto [start, end] = get_tile_range(tile_idx, num_items, num_tiles);

    Request tile_req;
    tile_req.op = op;
    tile_req.future_id = batch_future_id;
    tile_req.batch = batch_state;
    tile_req.batch_chunk_num_bytes = batch_chunk_num_bytes;
    tile_req.start_idx = start;

    for (size_t i = start; i < end; ++i) {
      tile_req.keys.push_back(keys[i]);
      tile_req.buf_ptrs.push_back(bufs[i]);
      tile_req.buf_lens.push_back(lens[i]);
    }

    return tile_req;
  }

  void enqueue_request(Request&& req) {
    const std::string& key = lane_key_for_op(req.op);
    auto it = lanes_.find(key);
    RequestQueue& queue =
        it != lanes_.end() ? it->second->queue : shared_queue_;
    {
      std::lock_guard<std::mutex> lk(queue.mu);
      if (is_read_op(req.op)) {
        queue.reads.push_back(std::move(req));
        queue.pending_reads.store(queue.reads.size(),
                                  std::memory_order_release);
      } else {
        queue.writes.push_back(std::move(req));
      }
    }
    queue.cv.notify_one();
  }

  // put the unfinished remainder of a write tile back at the FRONT of the
  // write class, so it resumes before writes that were submitted after it
  void requeue_yielded_write(RequestQueue& queue, Request&& req) {
    {
      std::lock_guard<std::mutex> lk(queue.mu);
      queue.writes.push_front(std::move(req));
    }
    queue.cv.notify_one();
  }

  // pop the next request according to the scheduling policy described at
  // the top of this file. caller must hold queue.mu and guarantee that at
  // least one of the two classes is non-empty
  static Request pop_next_request_locked(RequestQueue& queue) {
    bool take_write =
        queue.reads.empty() ||
        (!queue.writes.empty() && queue.read_streak >= kMaxReadStreak);
    Request req;
    if (take_write) {
      req = std::move(queue.writes.front());
      queue.writes.pop_front();
      queue.read_streak = 0;
    } else {
      req = std::move(queue.reads.front());
      queue.reads.pop_front();
      queue.pending_reads.store(queue.reads.size(),
                                std::memory_order_release);
      // only count reads that actually overtook a waiting write
      queue.read_streak = queue.writes.empty() ? 0 : queue.read_streak + 1;
    }
    return req;
  }

  void push_completion(Completion&& c) {
    {
      std::lock_guard<std::mutex> lk(comp_mu_);
      completions_.push(std::move(c));
    }
    signal_eventfd_();
  }

  void drain_eventfd_() { notifier_->consume(); }

  void signal_eventfd_() {
    bool already_signaled = signaled_.exchange(true, std::memory_order_acq_rel);
    if (already_signaled) return;  // only one signal at a time
    notifier_->notify();
  }

  static const std::unordered_map<std::string, std::vector<Op>>&
  lane_registry() {
    static const std::unordered_map<std::string, std::vector<Op>> registry = {
        {"lookup", {Op::BATCH_TILE_EXISTS}},
        {"retrieve", {Op::BATCH_TILE_GET}},
        {"store", {Op::BATCH_TILE_SET}},
        {"delete", {Op::BATCH_TILE_DELETE}},
    };
    return registry;
  }

  static const std::string& lane_key_for_op(Op op) {
    static const auto& reverse = []() {
      std::unordered_map<Op, std::string> m;
      for (auto& [key, ops] : lane_registry()) {
        for (Op o : ops) {
          m[o] = key;
        }
      }
      return m;
    }();
    auto it = reverse.find(op);
    if (it == reverse.end()) {
      throw std::runtime_error("unknown Op type: " +
                               std::to_string(static_cast<int>(op)));
    }
    return it->second;
  }

  void start_lane_workers(WorkerLane& lane, int num_workers) {
    lane.workers.reserve(static_cast<size_t>(num_workers));
    WorkerLane* lane_ptr = &lane;
    for (int i = 0; i < num_workers; i++) {
      lane.workers.emplace_back(
          [this, lane_ptr]() { this->worker_loop_for_queue(lane_ptr->queue); });
    }
  }

  void join_lane_workers(WorkerLane& lane) {
    for (auto& worker : lane.workers) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  void clear_queue(RequestQueue& queue) {
    std::lock_guard<std::mutex> lk(queue.mu);
    queue.reads.clear();
    queue.writes.clear();
    queue.pending_reads.store(0, std::memory_order_release);
  }

  void worker_loop() { worker_loop_for_queue(shared_queue_); }

  void worker_loop_for_queue(RequestQueue& queue) {
    // lets do_batch_set() ask, through yield_to_pending_read(), whether a
    // read is waiting on the queue this thread serves
    WorkerContext ctx;
    ctx.queue = &queue;
    worker_context_ = &ctx;

    try {
      // create connection (derived class specific)
      ConnectionType conn = create_connection();

      for (;;) {
        Request req;

        // 1. grab a request from the submission queue
        {
          std::unique_lock<std::mutex> lk(queue.mu);
          queue.cv.wait(lk, [&] {
            return stop_.load(std::memory_order_acquire) ||
                   !queue.reads.empty() || !queue.writes.empty();
          });
          if (queue.reads.empty() && queue.writes.empty()) {
            break;  // stopping and nothing left to do: exit loop
          }
          req = pop_next_request_locked(queue);
        }

        Completion comp;
        comp.future_id = req.future_id;
        ctx.yield_idx = kNotYielded;

        // 2. execute the requested operation
        try {
          switch (req.op) {
            case Op::BATCH_TILE_GET:
              do_batch_get(conn, req);
              comp.ok = true;
              break;

            case Op::BATCH_TILE_SET:
              do_batch_set(conn, req);
              comp.ok = true;
              break;

            case Op::BATCH_TILE_EXISTS:
              do_batch_exists(conn, req);
              comp.ok = true;
              break;

            case Op::BATCH_TILE_DELETE:
              do_batch_delete(conn, req);
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

        // 3. a write tile that yielded to a pending read is not finished:
        // put the remainder back and go serve the read. its batch state is
        // left untouched, so the batch completes once, after its last key
        if (comp.ok && ctx.yield_idx != kNotYielded) {
          req.resume_idx = ctx.yield_idx;
          requeue_yielded_write(queue, std::move(req));
          continue;
        }

        // 4. update shared batch state and push completion when done
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
      // for batch exists and batch get, move per-key results
      if (req.batch->batch_op == Op::BATCH_TILE_EXISTS ||
          req.batch->batch_op == Op::BATCH_TILE_GET ||
          req.batch->batch_op == Op::BATCH_TILE_DELETE) {
        batch_comp.result_bytes = std::move(req.batch->per_key_results);
      }
      push_completion(std::move(batch_comp));
    }
  }

 protected:
  int num_workers_;
  WorkerPoolConfig worker_pool_config_;

  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::atomic<uint64_t> next_future_id_{1};

 private:
  std::unique_ptr<EventNotifier> notifier_;

  // treat wakeup fd as a binary wakeup flag:
  // true: Python has been signaled (or will be)
  // false: Python is asleep, no wakeup pending
  std::atomic<bool> signaled_{false};

  // submission queue (SQ) of the shared worker pool
  RequestQueue shared_queue_;

  // context of the worker running on the current thread; null on any
  // thread that is not one of this class's workers
  static inline thread_local WorkerContext* worker_context_ = nullptr;

  // completion queue (CQ)
  std::mutex comp_mu_;
  std::queue<Completion> completions_;

  std::vector<std::thread> workers_;
  // Populated only during start_workers() (single-threaded construction).
  // All subsequent accesses are reads — no lock needed in enqueue_request.
  std::unordered_map<std::string, std::unique_ptr<WorkerLane>> lanes_;
};

}  // namespace connector
}  // namespace lmcache
