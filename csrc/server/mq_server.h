// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — ZMQ message queue server
//
// Mirrors Python MessageQueueServer: ZMQ ROUTER socket with eventfd
// notification from thread pool callbacks to the main poll loop.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "types.h"

namespace lmcache {
namespace server {

// ============================================================================
// IRequestHandler — abstract handler interface
// ============================================================================

class IRequestHandler {
 public:
  virtual ~IRequestHandler() = default;

  /// Handle a synchronous request (called in the main loop).
  /// @param payloads  Raw msgpack payload frames
  /// @return          Encoded response bytes (empty = no response)
  virtual std::vector<uint8_t> handle_sync(
      const std::vector<std::vector<uint8_t>>& payloads) = 0;

  /// Handle a blocking request (called on a thread pool worker).
  /// @param payloads  Raw msgpack payload frames
  /// @return          Encoded response bytes (empty = no response)
  virtual std::vector<uint8_t> handle_blocking(
      const std::vector<std::vector<uint8_t>>& payloads) = 0;

  /// Whether this handler runs synchronously or on a thread pool.
  virtual HandlerType handler_type() const = 0;
};

// ============================================================================
// MessageQueueServer — ZMQ ROUTER + eventfd + thread pool
// ============================================================================

class MessageQueueServer {
 public:
  /// @param bind_url     ZMQ bind URL (e.g. "tcp://0.0.0.0:8001")
  /// @param max_workers  Thread pool size for blocking handlers
  MessageQueueServer(const std::string& bind_url, int max_workers = 4);

  ~MessageQueueServer();

  // Non-copyable
  MessageQueueServer(const MessageQueueServer&) = delete;
  MessageQueueServer& operator=(const MessageQueueServer&) = delete;

  /// Register a handler for a request type.
  void add_handler(RequestType type, std::unique_ptr<IRequestHandler> handler);

  /// Assign an affinity thread pool to specific request types.
  /// Requests from the same ZMQ identity are always dispatched to the same
  /// worker thread (hash identity → worker index). Use for GPU-bound handlers.
  /// Must be called after add_handler() and before start().
  void add_affinity_thread_pool(const std::vector<RequestType>& request_types,
                                int max_workers);

  /// Assign a normal (round-robin) thread pool to specific request types.
  /// Use for non-GPU blocking handlers.
  /// Must be called after add_handler() and before start().
  void add_normal_thread_pool(const std::vector<RequestType>& request_types,
                              int max_workers);

  /// Start the server (spawns the main loop thread).
  void start();

  /// Stop the server and join threads.
  void close();

 private:
  /// Main event loop: zmq_poll over socket fd + eventfd.
  void main_loop();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace server
}  // namespace lmcache
