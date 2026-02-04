// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>

namespace py = pybind11;

class MultiRESPClient {
 public:
  MultiRESPClient(std::string host, int port, size_t chunk_size,
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
};
