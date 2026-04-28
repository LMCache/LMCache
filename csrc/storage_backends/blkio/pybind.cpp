// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>

#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_blkio, m) {
  py::class_<lmcache::connector::BlkioConnector>(m, "LMCacheBlkioClient")
      .def(py::init<std::string, int, bool>(), py::arg("device_path"),
           py::arg("num_workers"), py::arg("direct_io") = true)
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::BlkioConnector);
}
