// SPDX-License-Identifier: Apache-2.0
#include <stdexcept>
#include <string>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

namespace {

int parse_optional_worker_count(const py::object& value, const char* arg_name) {
  if (value.is_none()) {
    return 0;
  }
  int count = value.cast<int>();
  if (count <= 0) {
    throw std::runtime_error(std::string(arg_name) +
                             " must be a positive integer or None");
  }
  return count;
}

}  // namespace

PYBIND11_MODULE(lmcache_mooncake, m) {
  py::class_<lmcache::connector::L1RegistrationConfig>(m,
                                                       "L1RegistrationConfig")
      .def(py::init<>())
      .def_readwrite("enabled",
                     &lmcache::connector::L1RegistrationConfig::enabled)
      .def_readwrite("base", &lmcache::connector::L1RegistrationConfig::base)
      .def_readwrite("size", &lmcache::connector::L1RegistrationConfig::size);

  py::class_<lmcache::connector::MooncakeConnector>(m, "LMCacheMooncakeClient")
      .def(py::init([](lmcache::connector::ConfigDict config, int num_workers,
                       lmcache::connector::L1RegistrationConfig l1_registration,
                       py::object lookup_workers, py::object retrieve_workers,
                       py::object store_workers) {
             return new lmcache::connector::MooncakeConnector(
                 std::move(config), num_workers, l1_registration,
                 parse_optional_worker_count(lookup_workers, "lookup_workers"),
                 parse_optional_worker_count(retrieve_workers,
                                             "retrieve_workers"),
                 parse_optional_worker_count(store_workers, "store_workers"));
           }),
           py::arg("config"), py::arg("num_workers"),
           py::arg("l1_registration") =
               lmcache::connector::L1RegistrationConfig{},
           py::arg("lookup_workers") = py::none(),
           py::arg("retrieve_workers") = py::none(),
           py::arg("store_workers") = py::none())
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::MooncakeConnector);
}
