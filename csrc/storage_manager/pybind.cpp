// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include "ttl_lock.h"

namespace py = pybind11;

using lmcache::storage_manager::TTLLock;

PYBIND11_MODULE(native_storage_ops, m) {
  m.doc() = "Native storage operations for LMCache";

  py::class_<TTLLock>(m, "TTLLock")
      .def(py::init<uint32_t>(), py::arg("ttl_second") = 300,
           "Construct a TTLLock with the specified TTL duration in "
           "seconds. Default is 300 seconds.")
      .def("lock", &TTLLock::lock,
           "Increment the lock counter by 1 and update the TTL. "
           "If the previous TTL has expired, reset counter to 1.")
      .def("unlock", &TTLLock::unlock,
           "Decrement the lock counter by 1 (minimum 0).")
      .def("is_locked", &TTLLock::is_locked,
           "Check if the lock is held (counter > 0 and TTL not expired).")
      .def("reset", &TTLLock::reset,
           "Reset the lock to initial state (counter = 0, TTL expired).");
}
