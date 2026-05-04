// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_mooncake, m) {
  py::class_<lmcache::connector::L1RegistrationConfig>(m,
                                                       "L1RegistrationConfig")
      .def(py::init<>())
      .def_readwrite("enabled",
                     &lmcache::connector::L1RegistrationConfig::enabled)
      .def_readwrite("base", &lmcache::connector::L1RegistrationConfig::base)
      .def_readwrite("size", &lmcache::connector::L1RegistrationConfig::size);

  py::class_<lmcache::connector::MooncakeConnector>(m, "LMCacheMooncakeClient")
      .def(py::init<lmcache::connector::ConfigDict, int,
                    lmcache::connector::L1RegistrationConfig>(),
           py::arg("config"), py::arg("num_workers"),
           py::arg("l1_registration") =
               lmcache::connector::L1RegistrationConfig{})
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::MooncakeConnector);
}
