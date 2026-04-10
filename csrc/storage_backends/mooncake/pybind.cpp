// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_mooncake, m) {
  py::class_<lmcache::connector::MooncakeConnector>(m, "LMCacheMooncakeClient")
      .def(py::init<lmcache::connector::ConfigDict, int, std::uintptr_t,
                    size_t>(),
           py::arg("config"), py::arg("num_workers"),
           py::arg("preregister_l1_base") = 0,
           py::arg("preregister_l1_size") = 0)
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::MooncakeConnector);
}
