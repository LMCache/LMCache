// SPDX-License-Identifier: Apache-2.0
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_nixl, module) {
  // This module also ships as the separate lmcache-nixl package, so it can be
  // installed next to a different lmcache. Bump API_VERSION, and
  // NIXL_CONNECTOR_API_VERSION in nixl_native_l2_adapter.py, whenever the
  // Python API changes: the constructor below or
  // LMCACHE_BIND_CONNECTOR_METHODS.
  module.attr("API_VERSION") = 2;
  py::class_<lmcache::connector::NixlConnector>(module, "LMCacheNixlClient")
      .def(
          py::init<std::string, std::unordered_map<std::string, std::string>,
                   int, uintptr_t, size_t, size_t,
                   std::vector<std::unordered_map<std::string, std::string>>>(),
          py::arg("backend"), py::arg("backend_params"), py::arg("num_workers"),
          py::arg("l1_base"), py::arg("l1_size"), py::arg("l1_alignment"),
          py::arg("worker_backend_params") =
              std::vector<std::unordered_map<std::string, std::string>>())
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
