// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_nixl, module) {
  py::class_<lmcache::connector::NixlConnector>(module, "LMCacheNixlClient")
      .def(py::init<std::string, std::unordered_map<std::string, std::string>,
                    int, uintptr_t, size_t, size_t>(),
           py::arg("backend"), py::arg("backend_params"),
           py::arg("num_workers"), py::arg("l1_base"), py::arg("l1_size"),
           py::arg("l1_alignment"))
      .def_property_readonly("storage_type",
                             &lmcache::connector::NixlConnector::storage_type)
      .def_property_readonly("supports_query",
                             &lmcache::connector::NixlConnector::supports_query)
      .def_property_readonly(
          "supports_delete",
          &lmcache::connector::NixlConnector::supports_delete)
      .def_property_readonly(
          "supports_direct_io",
          &lmcache::connector::NixlConnector::supports_direct_io)
      .def_property_readonly(
          "atomic_publication",
          &lmcache::connector::NixlConnector::atomic_publication)
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::NixlConnector);
}
