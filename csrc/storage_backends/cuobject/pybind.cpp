// SPDX-License-Identifier: Apache-2.0
//
// pybind11 module for the cuObject client wrapper.
//
// Module name: lmcache_cuobject
// Exposes: CuObjectClient class and cuObject API constants.
//
// All methods that interact with the cuObject C library release the GIL
// so they can run concurrently with Python threads.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cuobject_client.h"

namespace py = pybind11;

PYBIND11_MODULE(lmcache_cuobject, m) {
  m.doc() = "LMCache cuObject RDMA client bindings (pybind11)";

  // Re-export constants so Python can access them without a separate header.
  m.attr("CU_OBJ_SUCCESS") = lmcache::cuobject::CU_OBJ_SUCCESS;
  m.attr("CU_OBJ_FAIL") = lmcache::cuobject::CU_OBJ_FAIL;
  m.attr("CUOBJ_PROTO_RDMA_DC_V1") = lmcache::cuobject::CUOBJ_PROTO_RDMA_DC_V1;
  m.attr("CUOBJ_MAX_MEMORY_REG_SIZE") =
      lmcache::cuobject::CUOBJ_MAX_MEMORY_REG_SIZE;

  // Memory type constants (for diagnostic use).
  m.attr("CUOBJ_MEMORY_SYSTEM") = lmcache::cuobject::CUOBJ_MEMORY_SYSTEM;
  m.attr("CUOBJ_MEMORY_CUDA_MANAGED") =
      lmcache::cuobject::CUOBJ_MEMORY_CUDA_MANAGED;
  m.attr("CUOBJ_MEMORY_CUDA_DEVICE") =
      lmcache::cuobject::CUOBJ_MEMORY_CUDA_DEVICE;

  py::class_<lmcache::cuobject::CuObjectClient>(m, "CuObjectClient")
      .def(py::init<int>(),
           py::arg("proto") = lmcache::cuobject::CUOBJ_PROTO_RDMA_DC_V1)
      .def("register_pool",
           &lmcache::cuobject::CuObjectClient::register_pool, py::arg("ptr"),
           py::arg("size"), py::call_guard<py::gil_scoped_release>())
      .def("deregister_pool",
           &lmcache::cuobject::CuObjectClient::deregister_pool,
           py::arg("ptr"), py::call_guard<py::gil_scoped_release>())
      .def("prepare_put", &lmcache::cuobject::CuObjectClient::prepare_put,
           py::arg("data_ptr"), py::arg("size"),
           py::call_guard<py::gil_scoped_release>())
      .def("prepare_get", &lmcache::cuobject::CuObjectClient::prepare_get,
           py::arg("data_ptr"), py::arg("size"),
           py::call_guard<py::gil_scoped_release>())
      .def("is_connected",
           &lmcache::cuobject::CuObjectClient::is_connected,
           py::call_guard<py::gil_scoped_release>())
      .def("close", &lmcache::cuobject::CuObjectClient::close,
           py::call_guard<py::gil_scoped_release>());
}
