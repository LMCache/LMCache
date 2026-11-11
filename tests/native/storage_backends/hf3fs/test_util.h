// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "csrc/storage_backends/connector_interface.h"
#include <poll.h>
#include <chrono>
#include <cstdint>
#include <stdexcept>

// Wait for the completion of `id`, the only operation in flight.
inline lmcache::connector::Completion wait_for_completion(
    lmcache::connector::IStorageConnector& connector, uint64_t id) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (std::chrono::steady_clock::now() < deadline) {
    for (auto& completion : connector.drain_completions()) {
      if (completion.future_id == id) return completion;
      throw std::runtime_error("unexpected completion");
    }
    pollfd pfd{connector.event_fd(), POLLIN, 0};
    ::poll(&pfd, 1, 10);
  }
  throw std::runtime_error("native task did not complete");
}

inline void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
