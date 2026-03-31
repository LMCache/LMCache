// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — ZMQ message queue server implementation

#include "mq_server.h"

#include <sys/eventfd.h>
#include <unistd.h>
#include <zmq.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

namespace lmcache {
namespace server {

// ============================================================================
// IThreadPool — abstract interface for thread pools
// ============================================================================

struct IThreadPool {
  virtual ~IThreadPool() = default;
  /// Submit a task, optionally with an affinity key for routing.
  virtual void submit(std::function<void()> task,
                      uint64_t affinity_key = 0) = 0;
};

// ============================================================================
// NormalThreadPool — round-robin worker pool
// ============================================================================

struct NormalThreadPool : IThreadPool {
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  std::mutex mutex;
  std::condition_variable cv;
  bool stop = false;

  explicit NormalThreadPool(int max_workers) {
    for (int i = 0; i < max_workers; ++i) {
      workers.emplace_back([this] { worker_loop(); });
    }
  }

  ~NormalThreadPool() override {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    cv.notify_all();
    for (auto& w : workers) {
      if (w.joinable()) w.join();
    }
  }

  void submit(std::function<void()> task, uint64_t /*affinity_key*/) override {
    {
      std::lock_guard<std::mutex> lock(mutex);
      tasks.push(std::move(task));
    }
    cv.notify_one();
  }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] { return stop || !tasks.empty(); });
        if (stop && tasks.empty()) return;
        task = std::move(tasks.front());
        tasks.pop();
      }
      task();
    }
  }
};

// ============================================================================
// AffinityThreadPool — routes tasks by affinity_key to fixed workers
// ============================================================================

struct AffinityThreadPool : IThreadPool {
  int num_workers_;
  struct WorkerQueue {
    std::queue<std::function<void()>> tasks;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop = false;
  };
  std::vector<std::unique_ptr<WorkerQueue>> queues_;
  std::vector<std::thread> workers_;

  explicit AffinityThreadPool(int max_workers) : num_workers_(max_workers) {
    queues_.reserve(max_workers);
    workers_.reserve(max_workers);
    for (int i = 0; i < max_workers; ++i) {
      queues_.push_back(std::make_unique<WorkerQueue>());
      auto* q = queues_.back().get();
      workers_.emplace_back([q] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock<std::mutex> lock(q->mutex);
            q->cv.wait(lock, [q] { return q->stop || !q->tasks.empty(); });
            if (q->stop && q->tasks.empty()) return;
            task = std::move(q->tasks.front());
            q->tasks.pop();
          }
          task();
        }
      });
    }
  }

  ~AffinityThreadPool() override {
    for (auto& q : queues_) {
      {
        std::lock_guard<std::mutex> lock(q->mutex);
        q->stop = true;
      }
      q->cv.notify_one();
    }
    for (auto& w : workers_) {
      if (w.joinable()) w.join();
    }
  }

  void submit(std::function<void()> task, uint64_t affinity_key) override {
    int slot = static_cast<int>(affinity_key % num_workers_);
    auto& q = queues_[slot];
    {
      std::lock_guard<std::mutex> lock(q->mutex);
      q->tasks.push(std::move(task));
    }
    q->cv.notify_one();
  }
};

// ============================================================================
// ZMQ multipart helpers
// ============================================================================

/// Receive a complete multipart message from a ZMQ socket.
/// Returns frames as vector<vector<uint8_t>>.  Returns empty on error.
static std::vector<std::vector<uint8_t>> zmq_recv_multipart(void* socket) {
  std::vector<std::vector<uint8_t>> frames;
  int more = 1;
  while (more) {
    zmq_msg_t msg;
    zmq_msg_init(&msg);
    int rc = zmq_msg_recv(&msg, socket, 0);
    if (rc < 0) {
      zmq_msg_close(&msg);
      return {};  // error
    }
    size_t size = zmq_msg_size(&msg);
    auto* data = static_cast<uint8_t*>(zmq_msg_data(&msg));
    frames.emplace_back(data, data + size);
    size_t more_size = sizeof(more);
    zmq_getsockopt(socket, ZMQ_RCVMORE, &more, &more_size);
    zmq_msg_close(&msg);
  }
  return frames;
}

/// Send a multipart message on a ZMQ socket.
static bool zmq_send_multipart(
    void* socket, const std::vector<std::vector<uint8_t>>& frames) {
  for (size_t i = 0; i < frames.size(); ++i) {
    int flags = (i + 1 < frames.size()) ? ZMQ_SNDMORE : 0;
    zmq_msg_t msg;
    zmq_msg_init_size(&msg, frames[i].size());
    std::memcpy(zmq_msg_data(&msg), frames[i].data(), frames[i].size());
    int rc = zmq_msg_send(&msg, socket, flags);
    if (rc < 0) {
      zmq_msg_close(&msg);
      return false;
    }
    // zmq_msg_send takes ownership on success — no close needed
  }
  return true;
}

// ============================================================================
// Decode RequestType from a msgpack-encoded int frame
// ============================================================================

/// RequestType values are 1–21, which map to msgpack positive fixint (single
/// byte 0x01–0x15).  For robustness we also handle uint8 (0xcc XX) and
/// uint16 (0xcd XX XX) formats.
static RequestType decode_request_type_raw(const uint8_t* data, size_t len) {
  if (len == 0) {
    return static_cast<RequestType>(0);
  }
  uint8_t head = data[0];
  // positive fixint: 0x00–0x7f
  if (head <= 0x7f) {
    return static_cast<RequestType>(head);
  }
  // uint8 format: 0xcc + 1 byte
  if (head == 0xcc && len >= 2) {
    return static_cast<RequestType>(data[1]);
  }
  // uint16 format: 0xcd + 2 bytes big-endian
  if (head == 0xcd && len >= 3) {
    uint16_t val = (static_cast<uint16_t>(data[1]) << 8) | data[2];
    return static_cast<RequestType>(val);
  }
  // negative fixint or other — shouldn't happen for valid request types
  return static_cast<RequestType>(0);
}

/// Compute a hash from a ZMQ identity frame for affinity routing.
static uint64_t identity_hash(const std::vector<uint8_t>& identity) {
  // FNV-1a hash
  uint64_t h = 14695981039346656037ULL;
  for (uint8_t b : identity) {
    h ^= b;
    h *= 1099511628211ULL;
  }
  return h;
}

// ============================================================================
// MessageQueueServer::Impl
// ============================================================================

struct MessageQueueServer::Impl {
  std::string bind_url;
  void* zmq_ctx = nullptr;
  void* zmq_socket = nullptr;
  int output_efd = -1;

  std::thread main_thread;
  std::atomic<bool> is_finished{false};

  // Owned pools
  std::vector<std::unique_ptr<IThreadPool>> pools;

  // Handlers
  std::unordered_map<int, std::unique_ptr<IRequestHandler>> handlers;

  // Per-handler pool assignment (request type int → pool pointer)
  std::unordered_map<int, IThreadPool*> handler_pools;

  // Whether a handler uses affinity routing
  std::unordered_map<int, bool> handler_uses_affinity;

  // Output queue for responses from blocking handlers
  std::queue<std::vector<std::vector<uint8_t>>> output_queue;
  std::mutex output_mutex;
};

// ============================================================================
// MessageQueueServer — constructor / destructor
// ============================================================================

MessageQueueServer::MessageQueueServer(const std::string& bind_url,
                                       int /*max_workers*/)
    : impl_(std::make_unique<Impl>()) {
  impl_->bind_url = bind_url;

  // Create eventfd for thread→main notification
  impl_->output_efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
  if (impl_->output_efd < 0) {
    throw std::runtime_error("Failed to create eventfd: " +
                             std::string(strerror(errno)));
  }

  // Create ZMQ context and ROUTER socket
  impl_->zmq_ctx = zmq_ctx_new();
  if (!impl_->zmq_ctx) {
    ::close(impl_->output_efd);
    throw std::runtime_error("Failed to create ZMQ context");
  }

  impl_->zmq_socket = zmq_socket(impl_->zmq_ctx, ZMQ_ROUTER);
  if (!impl_->zmq_socket) {
    zmq_ctx_destroy(impl_->zmq_ctx);
    ::close(impl_->output_efd);
    throw std::runtime_error("Failed to create ZMQ ROUTER socket");
  }

  int rc = zmq_bind(impl_->zmq_socket, bind_url.c_str());
  if (rc != 0) {
    zmq_close(impl_->zmq_socket);
    zmq_ctx_destroy(impl_->zmq_ctx);
    ::close(impl_->output_efd);
    throw std::runtime_error("Failed to bind ZMQ socket to " + bind_url + ": " +
                             zmq_strerror(zmq_errno()));
  }

  std::cerr << "[MQServer] Bound to " << bind_url << std::endl;
}

MessageQueueServer::~MessageQueueServer() { close(); }

// ============================================================================
// add_handler / add_affinity_thread_pool / add_normal_thread_pool / start /
// close
// ============================================================================

void MessageQueueServer::add_handler(RequestType type,
                                     std::unique_ptr<IRequestHandler> handler) {
  int key = static_cast<int>(type);
  impl_->handlers[key] = std::move(handler);
}

void MessageQueueServer::add_affinity_thread_pool(
    const std::vector<RequestType>& request_types, int max_workers) {
  if (request_types.empty()) return;

  auto pool = std::make_unique<AffinityThreadPool>(max_workers);
  IThreadPool* pool_ptr = pool.get();

  for (auto rt : request_types) {
    int key = static_cast<int>(rt);
    auto it = impl_->handlers.find(key);
    if (it == impl_->handlers.end()) {
      throw std::runtime_error(
          "add_affinity_thread_pool: no handler registered for request type " +
          std::to_string(key));
    }
    if (it->second->handler_type() != HandlerType::BLOCKING) {
      throw std::runtime_error(
          "add_affinity_thread_pool: handler for request type " +
          std::to_string(key) + " is not BLOCKING");
    }
    impl_->handler_pools[key] = pool_ptr;
    impl_->handler_uses_affinity[key] = true;
  }

  impl_->pools.push_back(std::move(pool));
  std::cerr << "[MQServer] Created affinity thread pool (max_workers="
            << max_workers << ") for " << request_types.size()
            << " request type(s)" << std::endl;
}

void MessageQueueServer::add_normal_thread_pool(
    const std::vector<RequestType>& request_types, int max_workers) {
  if (request_types.empty()) return;

  auto pool = std::make_unique<NormalThreadPool>(max_workers);
  IThreadPool* pool_ptr = pool.get();

  for (auto rt : request_types) {
    int key = static_cast<int>(rt);
    auto it = impl_->handlers.find(key);
    if (it == impl_->handlers.end()) {
      throw std::runtime_error(
          "add_normal_thread_pool: no handler registered for request type " +
          std::to_string(key));
    }
    if (it->second->handler_type() != HandlerType::BLOCKING) {
      throw std::runtime_error(
          "add_normal_thread_pool: handler for request type " +
          std::to_string(key) + " is not BLOCKING");
    }
    impl_->handler_pools[key] = pool_ptr;
    impl_->handler_uses_affinity[key] = false;
  }

  impl_->pools.push_back(std::move(pool));
  std::cerr << "[MQServer] Created normal thread pool (max_workers="
            << max_workers << ") for " << request_types.size()
            << " request type(s)" << std::endl;
}

void MessageQueueServer::start() {
  // Validate all blocking handlers have a pool assigned
  for (auto& [rt_int, handler] : impl_->handlers) {
    if (handler->handler_type() == HandlerType::BLOCKING) {
      if (impl_->handler_pools.find(rt_int) == impl_->handler_pools.end()) {
        throw std::runtime_error(
            "BlockingHandler for request type " + std::to_string(rt_int) +
            " has no thread pool assigned. Call add_affinity_thread_pool or "
            "add_normal_thread_pool before start().");
      }
    }
  }

  impl_->main_thread = std::thread([this] { main_loop(); });
  std::cerr << "[MQServer] Started main loop thread" << std::endl;
}

void MessageQueueServer::close() {
  if (impl_->is_finished.exchange(true)) return;  // already closed

  // Join main loop thread
  if (impl_->main_thread.joinable()) {
    impl_->main_thread.join();
  }

  // Close ZMQ socket and context
  if (impl_->zmq_socket) {
    zmq_close(impl_->zmq_socket);
    impl_->zmq_socket = nullptr;
  }
  if (impl_->zmq_ctx) {
    zmq_ctx_destroy(impl_->zmq_ctx);
    impl_->zmq_ctx = nullptr;
  }

  // Close eventfd
  if (impl_->output_efd >= 0) {
    ::close(impl_->output_efd);
    impl_->output_efd = -1;
  }

  // Shutdown all owned pools (destructors join worker threads)
  impl_->pools.clear();

  std::cerr << "[MQServer] Closed" << std::endl;
}

// ============================================================================
// main_loop — zmq_poll over ROUTER socket + eventfd
// ============================================================================

void MessageQueueServer::main_loop() {
  // Get the ZMQ socket fd for polling alongside eventfd
  int zmq_fd = -1;
  size_t zmq_fd_size = sizeof(zmq_fd);
  zmq_getsockopt(impl_->zmq_socket, ZMQ_FD, &zmq_fd, &zmq_fd_size);

  while (!impl_->is_finished.load(std::memory_order_relaxed)) {
    // zmq_poll with two items: the ROUTER socket and the eventfd
    zmq_pollitem_t items[2];
    items[0].socket = impl_->zmq_socket;
    items[0].fd = 0;
    items[0].events = ZMQ_POLLIN;
    items[0].revents = 0;
    items[1].socket = nullptr;
    items[1].fd = impl_->output_efd;
    items[1].events = ZMQ_POLLIN;
    items[1].revents = 0;

    int rc = zmq_poll(items, 2, 1000 /* ms */);
    if (rc < 0) {
      if (zmq_errno() == EINTR) continue;
      std::cerr << "[MQServer] zmq_poll error: " << zmq_strerror(zmq_errno())
                << std::endl;
      break;
    }

    // --- Handle inbound requests on the ROUTER socket ---
    if (items[0].revents & ZMQ_POLLIN) {
      // ZMQ edge-triggered: drain all available messages
      while (true) {
        auto frames = zmq_recv_multipart(impl_->zmq_socket);
        if (frames.empty()) break;  // EAGAIN or error

        // Need at least 3 frames: [identity, request_uid, request_type, ...]
        if (frames.size() < 3) {
          std::cerr << "[MQServer] Received message with too few frames ("
                    << frames.size() << ")" << std::endl;
          continue;
        }

        // Parse identity, request_uid, request_type
        auto identity = std::move(frames[0]);
        auto b_request_uid = std::move(frames[1]);
        auto b_request_type = std::move(frames[2]);

        // Collect payload frames
        std::vector<std::vector<uint8_t>> payloads;
        for (size_t i = 3; i < frames.size(); ++i) {
          payloads.push_back(std::move(frames[i]));
        }

        // Decode request type
        RequestType request_type = decode_request_type_raw(
            b_request_type.data(), b_request_type.size());
        int rt_int = static_cast<int>(request_type);

        // Look up handler
        auto handler_it = impl_->handlers.find(rt_int);
        if (handler_it == impl_->handlers.end()) {
          std::cerr << "[MQServer] No handler for request type " << rt_int
                    << std::endl;
          continue;
        }

        IRequestHandler* handler = handler_it->second.get();

        // Build prefix frames for the response: [identity, request_uid,
        // request_type]
        std::vector<std::vector<uint8_t>> prefix_frames;
        // Save identity for affinity hashing before moving
        uint64_t id_hash = identity_hash(identity);
        prefix_frames.push_back(std::move(identity));
        prefix_frames.push_back(std::move(b_request_uid));
        prefix_frames.push_back(std::move(b_request_type));

        if (handler->handler_type() == HandlerType::SYNC) {
          // ---- SYNC: handle inline and send response immediately ----
          try {
            auto response = handler->handle_sync(payloads);
            auto reply = prefix_frames;
            if (!response.empty()) {
              reply.push_back(std::move(response));
            }
            zmq_send_multipart(impl_->zmq_socket, reply);
          } catch (const std::exception& e) {
            std::cerr << "[MQServer] Error in sync handler for type " << rt_int
                      << ": " << e.what() << std::endl;
          }
        } else {
          // ---- BLOCKING: submit to assigned thread pool ----
          auto shared_payloads =
              std::make_shared<std::vector<std::vector<uint8_t>>>(
                  std::move(payloads));
          auto shared_prefix =
              std::make_shared<std::vector<std::vector<uint8_t>>>(
                  std::move(prefix_frames));
          int efd = impl_->output_efd;
          auto* output_queue = &impl_->output_queue;
          auto* output_mutex = &impl_->output_mutex;

          // Look up assigned pool
          auto pool_it = impl_->handler_pools.find(rt_int);
          if (pool_it == impl_->handler_pools.end()) {
            std::cerr << "[MQServer] No pool assigned for blocking type "
                      << rt_int << std::endl;
            continue;
          }
          IThreadPool* pool = pool_it->second;

          pool->submit(
              [handler, shared_payloads, shared_prefix, efd, output_queue,
               output_mutex]() {
                try {
                  auto response = handler->handle_blocking(*shared_payloads);
                  auto reply = *shared_prefix;
                  if (!response.empty()) {
                    reply.push_back(std::move(response));
                  }

                  {
                    std::lock_guard<std::mutex> lock(*output_mutex);
                    output_queue->push(std::move(reply));
                  }

                  // Notify main loop via eventfd
                  uint64_t val = 1;
                  [[maybe_unused]] auto written =
                      ::write(efd, &val, sizeof(val));
                } catch (const std::exception& e) {
                  std::cerr
                      << "[MQServer] Error in blocking handler: " << e.what()
                      << std::endl;
                }
              },
              id_hash);
        }

        // Check if more messages are available (ZMQ level-triggered within
        // a single poll wakeup). Use ZMQ_EVENTS to check.
        int events = 0;
        size_t events_size = sizeof(events);
        zmq_getsockopt(impl_->zmq_socket, ZMQ_EVENTS, &events, &events_size);
        if (!(events & ZMQ_POLLIN)) break;
      }
    }

    // --- Handle outbound responses (eventfd readable) ---
    if (items[1].revents & ZMQ_POLLIN) {
      // Consume the eventfd counter
      uint64_t val;
      [[maybe_unused]] auto rd = ::read(impl_->output_efd, &val, sizeof(val));

      // Drain the output queue
      std::lock_guard<std::mutex> lock(impl_->output_mutex);
      while (!impl_->output_queue.empty()) {
        auto reply = std::move(impl_->output_queue.front());
        impl_->output_queue.pop();
        zmq_send_multipart(impl_->zmq_socket, reply);
      }
    }
  }
}

}  // namespace server
}  // namespace lmcache
