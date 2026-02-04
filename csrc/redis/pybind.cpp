// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "resp.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_redis, m) {
  py::class_<MultiRESPClient>(m, "LMCacheRedisClient")
      .def(py::init<std::string, int, size_t, int>(), py::arg("host"),
           py::arg("port"), py::arg("chunk_size"), py::arg("num_workers"))
      .def("event_fd", &MultiRESPClient::event_fd)
      .def("submit_get", &MultiRESPClient::submit_get, py::arg("key"),
           py::arg("memoryview"))
      .def("submit_set", &MultiRESPClient::submit_set, py::arg("key"),
           py::arg("memoryview"))
      .def("submit_exists", &MultiRESPClient::submit_exists, py::arg("key"))
      .def("submit_batch_get", &MultiRESPClient::submit_batch_get,
           py::arg("keys"), py::arg("memoryviews"))
      .def("submit_batch_set", &MultiRESPClient::submit_batch_set,
           py::arg("keys"), py::arg("memoryviews"))
      .def("drain_completions", &MultiRESPClient::drain_completions)
      .def("close", &MultiRESPClient::close);
}
