// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Samsung Electronics Co., Ltd.All Rights Reserved
// Authors: Wenwen Chen <wenwen.chen@samsung.com>

#include <pybind11/pybind11.h>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

/**
 * Python bindings for Hf3fsConnector.
 *
 * Exposes the native C++ 3FS connector to Python as LMCacheHf3fsClient.
 * Uses the LMCACHE_BIND_CONNECTOR_METHODS macro to bind standard connector
 * methods.
 *
 * Example usage:
 * @code
 *   from lmcache.lmcache_hf3fs import LMCacheHf3fsClient
 *
 *   client = LMCacheHf3fsClient(
 *       mount_point="/mnt/3fs",
 *       base_paths="/mnt/3fs/path1,/mnt/3fs/path2",
 *       num_workers=8,
 *       ior_entries=256,
 *       io_depth=0,
 *       numa_id=-1,
 *       iov_size=209715200,
 *   )
 * @endcode
 */
PYBIND11_MODULE(lmcache_hf3fs, m) {
  py::class_<lmcache::connector::Hf3fsConnector>(m, "LMCacheHf3fsClient")
      .def(py::init([](std::string mount_point, std::string base_paths,
                       int num_workers, int ior_entries, int io_depth,
                       int numa_id, size_t iov_size, int time_out,
                       bool enable_key_buffer, py::object per_op_workers) {
             return new lmcache::connector::Hf3fsConnector(
                 std::move(mount_point), std::move(base_paths), num_workers,
                 ior_entries, io_depth, numa_id, iov_size, time_out,
                 enable_key_buffer,
                 lmcache::connector::pybind_utils::parse_per_op_workers(
                     per_op_workers));
           }),
           py::arg("mount_point"), py::arg("base_paths"),
           py::arg("num_workers") = 8, py::arg("ior_entries") = 256,
           py::arg("io_depth") = 0, py::arg("numa_id") = -1,
           py::arg("iov_size") = 209715200, py::arg("time_out") = 200,
           py::arg("enable_key_buffer") = true,
           py::arg("per_op_workers") = py::none())
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::Hf3fsConnector);
}
