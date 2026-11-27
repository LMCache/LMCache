// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <utility>

#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_fs, m) {
  py::class_<lmcache::connector::FSConnector>(m, "LMCacheFSClient")
      .def(py::init([](std::string base_path, int num_workers,
                       std::string relative_tmp_dir, bool use_odirect,
                       size_t read_ahead_size, int num_read_workers,
                       size_t read_max_bytes_in_flight,
                       py::object per_op_workers) {
             return new lmcache::connector::FSConnector(
                 std::move(base_path), num_workers,
                 std::move(relative_tmp_dir), use_odirect, read_ahead_size,
                 num_read_workers, read_max_bytes_in_flight,
                 lmcache::connector::pybind_utils::parse_per_op_workers(
                     per_op_workers));
           }),
           py::arg("base_path"), py::arg("num_workers"),
           py::arg("relative_tmp_dir") = "", py::arg("use_odirect") = false,
           py::arg("read_ahead_size") = 0, py::arg("num_read_workers") = 0,
           py::arg("read_max_bytes_in_flight") = 0,
           py::arg("per_op_workers") = py::none())
      .def("read_budget_bytes",
           &lmcache::connector::FSConnector::read_budget_bytes,
           "Bytes allowed outstanding against the device; 0 when there is "
           "no read pool.")
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::FSConnector);
}
