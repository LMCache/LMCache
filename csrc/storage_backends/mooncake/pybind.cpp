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

lmcache::connector::WorkerPoolConfig parse_per_op_workers(
    const py::object& obj) {
  if (obj.is_none()) {
    return {};
  }
  py::dict d = obj.cast<py::dict>();
  std::unordered_map<std::string, int> per_op;
  for (auto& [key, value] : d) {
    std::string key_str = key.cast<std::string>();
    int count = value.cast<int>();
    if (count <= 0) {
      throw std::runtime_error("per_op_workers['" + key_str +
                               "'] must be a positive integer");
    }
    per_op[key_str] = count;
  }
  return {std::move(per_op)};
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
                       py::object per_op_workers) {
             return new lmcache::connector::MooncakeConnector(
                 std::move(config), num_workers, l1_registration,
                 parse_per_op_workers(per_op_workers));
           }),
           py::arg("config"), py::arg("num_workers"),
           py::arg("l1_registration") =
               lmcache::connector::L1RegistrationConfig{},
           py::arg("per_op_workers") = py::none())
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::MooncakeConnector);
}
