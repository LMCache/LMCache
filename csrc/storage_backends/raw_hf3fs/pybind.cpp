// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

// pybind11 bindings for LMCache3FSClient

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/buffer_info.h>
#include <map>

#include "hf3fs_connector.h"

namespace py = pybind11;
using namespace lmcache::connector;

PYBIND11_MODULE(lmcache_raw3fs, m) {
  m.doc() = "LMCache 3FS Connector — Python bindings for the C++ backend";

  // ── Completion ──────────────────────────────────────────────────────────
  py::class_<Completion>(m, "Completion")
      .def_readonly("future_id", &Completion::future_id)
      .def_readonly("ok", &Completion::ok)
      .def_readonly("result_bytes", &Completion::result_bytes)
      .def_readonly("error", &Completion::error);

  // ── Config ──────────────────────────────────────────────────────────────
  py::class_<Hf3fsConnector>(m, "HF3fsConnector")
      .def(py::init([](int num_workers, const std::string& meta_addr,
                       const std::string& mgmtd_addr, uint32_t chunk_size,
                       uint32_t transport_pool_max_connections,
                       size_t max_batch_bytes, uint32_t max_concurrent_requests,
                       uint32_t max_concurrent_requests_per_server,
                       const std::map<std::string, int>& per_op_workers,
                       const std::string& cluster_id, uint32_t chain_table_id,
                       const std::string& base_dir, bool force_use_tcp,
                       uint32_t dir_permission, uint32_t file_permission,
                       const std::string& token) {
             return new Hf3fsConnector(
                 num_workers, meta_addr, mgmtd_addr, chunk_size,
                 transport_pool_max_connections, max_batch_bytes,
                 max_concurrent_requests, max_concurrent_requests_per_server,
                 per_op_workers, cluster_id, chain_table_id, base_dir,
                 force_use_tcp, dir_permission, file_permission, token);
           }),
           py::arg("num_workers") = 4,
           py::arg("meta_addr") = "RDMA://192.168.1.100:8000",
           py::arg("mgmtd_addr") = "RDMA://192.168.1.100:8000",
           py::arg("chunk_size") = 4 * 1024 * 1024,
           py::arg("transport_pool_max_connections") = 8,
           py::arg("max_batch_bytes") = 32 * 1024 * 1024,
           py::arg("max_concurrent_requests") = 256,
           py::arg("max_concurrent_requests_per_server") = 64,
           py::arg("per_op_workers") = std::map<std::string, int>(),
           py::arg("cluster_id") = "stage", py::arg("chain_table_id") = 1,
           py::arg("base_dir") = "lmcache", py::arg("force_use_tcp") = false,
           py::arg("dir_permission") = 0755u,
           py::arg("file_permission") = 0644u, py::arg("token") = "",
           "Create a new HF3fsConnector instance.\n\n"
           "Args:\n"
           "    num_workers: Number of worker threads (default 4).\n"
           "    meta_addr: Meta server address.\n"
           "    mgmtd_addr: Mgmtd server address.\n"
           "    chunk_size: Chunk size in bytes (default 4MB; must be a "
           "multiple of 4MB).\n"
           "    transport_pool_max_connections: Max connections per IO worker "
           "(default 8, increase for better concurrency).\n"
           "    max_batch_bytes: Max bytes per StorageClient "
           "batchRead/batchWrite RPC (default 32MB).\n"
           "    max_concurrent_requests: Max concurrent StorageClient "
           "read/write requests (default 256).\n"
           "    max_concurrent_requests_per_server: Max concurrent "
           "StorageClient requests per storage node (default 64; 0 = 3FS "
           "default 8).\n"
           "    per_op_workers: Map from op lane "
           "(lookup/retrieve/store/delete) to dedicated worker count. Lanes "
           "not listed share the num_workers pool (default {}).\n"
           "    cluster_id: 3FS cluster identifier (default 'stage').\n"
           "    chain_table_id: ChainTableId used for the storage layout "
           "(default 1).\n"
           "    base_dir: 3FS root-relative KV directory (default 'lmcache').\n"
           "    force_use_tcp: Force TCP transport instead of RDMA (default "
           "false = RDMA).\n"
           "    dir_permission: Permission for the base_dir directory (default "
           "0755).\n"
           "    file_permission: Permission for created data_*_store files "
           "(default "
           "0644).\n"
           "    token: 3FS user token (default ''). Required only when the "
           "meta server enables `authenticate`; must match the process "
           "uid/gid.\n")
      .def("close", &Hf3fsConnector::close,
           "Close the connector and release all resources.")
      .def("cleanup_data_files", &Hf3fsConnector::cleanup_data_files,
           "Release storage chunks and delete 3FS data files from "
           "MetaClient.\n\n"
           "Used to reclaim disk space on 3FS nodes before test cleanup.\n")
      .def("print_timing_stats", &Hf3fsConnector::print_timing_stats,
           "Print aggregated hot-path timing stats to stderr.")
      .def("event_fd", &Hf3fsConnector::event_fd,
           "Return the event file descriptor for polling.")
      .def(
          "drain_completions",
          [](Hf3fsConnector& self) {
            // Convert each Completion to the (future_id, ok, error,
            // result_bools) tuple convention used by the shared
            // NativeConnectorL2Adapter demux loop, matching
            // LMCACHE_BIND_CONNECTOR_METHODS in connector_pybind_utils.h. Runs
            // under the GIL: drain_completions clears the Python buffer
            // keepalive map (decref'ing py::object), which requires the GIL.
            std::vector<Completion> completions = self.drain_completions();
            py::list result;
            for (auto& c : completions) {
              py::object bools_obj = py::none();
              if (!c.result_bytes.empty()) {
                py::list bools_list;
                for (uint8_t b : c.result_bytes) {
                  bools_list.append(bool(b));
                }
                bools_obj = bools_list;
              }
              result.append(
                  py::make_tuple(c.future_id, c.ok, c.error, bools_obj));
            }
            return result;
          },
          "Drain all available completions.\n\n"
          "Returns:\n"
          "    list of (future_id, ok, error, result_bools) tuples.\n")
      .def(
          "submit_batch_get",
          [](Hf3fsConnector& self, const std::vector<std::string>& keys,
             py::list memviews) {
            if (keys.size() != memviews.size()) {
              throw std::runtime_error("keys and memviews size mismatch");
            }
            if (keys.empty()) {
              throw std::runtime_error("keys list is empty");
            }
            std::vector<void*> bufs;
            std::vector<size_t> lens;
            bufs.reserve(keys.size());
            lens.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) {
              py::memoryview mv = memviews[i].cast<py::memoryview>();
              py::buffer_info info = py::buffer(mv).request();
              bufs.push_back(info.ptr);
              lens.push_back(static_cast<size_t>(info.size));
            }
            size_t batch_chunk_num_bytes = lens[0];

            uint64_t future_id;
            {
              py::gil_scoped_release release;
              future_id = self.submit_batch_get(keys, bufs, lens,
                                                batch_chunk_num_bytes);
            }
            // Keep the memview objects alive until drain_completions consumes
            // this future_id, since the C++ workers read/write them
            // asynchronously.
            auto& ka = self.get_python_buffer_keepalive();
            ka.emplace(future_id, std::vector<py::object>());
            for (size_t i = 0; i < keys.size(); ++i) {
              ka.at(future_id).push_back(
                  py::reinterpret_borrow<py::object>(memviews[i]));
            }
            return future_id;
          },
          py::arg("keys"), py::arg("memoryviews"),
          "Submit a batch GET operation.\n\n"
          "Args:\n"
          "    keys: list of key strings to retrieve.\n"
          "    memoryviews: list of writable memoryviews, one per key.\n\n"
          "Returns:\n"
          "    int: future id for tracking this batch operation.\n")
      .def(
          "submit_batch_set",
          [](Hf3fsConnector& self, const std::vector<std::string>& keys,
             py::list memviews) {
            if (keys.size() != memviews.size()) {
              throw std::runtime_error("keys and memviews size mismatch");
            }
            if (keys.empty()) {
              throw std::runtime_error("keys list is empty");
            }
            std::vector<void*> bufs;
            std::vector<size_t> lens;
            bufs.reserve(keys.size());
            lens.reserve(keys.size());
            for (size_t i = 0; i < keys.size(); ++i) {
              py::memoryview mv = memviews[i].cast<py::memoryview>();
              py::buffer_info info = py::buffer(mv).request();
              bufs.push_back(info.ptr);
              lens.push_back(static_cast<size_t>(info.size));
            }
            size_t batch_chunk_num_bytes = lens[0];

            uint64_t future_id;
            {
              py::gil_scoped_release release;
              future_id = self.submit_batch_set(keys, bufs, lens,
                                                batch_chunk_num_bytes);
            }
            // Keep the memview objects alive until drain_completions consumes
            // this future_id, since the C++ workers read them asynchronously.
            auto& ka = self.get_python_buffer_keepalive();
            ka.emplace(future_id, std::vector<py::object>());
            for (size_t i = 0; i < keys.size(); ++i) {
              ka.at(future_id).push_back(
                  py::reinterpret_borrow<py::object>(memviews[i]));
            }
            return future_id;
          },
          py::arg("keys"), py::arg("memoryviews"),
          "Submit a batch SET operation.\n\n"
          "Args:\n"
          "    keys: list of key strings to store.\n"
          "    memoryviews: list of memoryviews containing each key's data.\n\n"
          "Returns:\n"
          "    int: future id for tracking this batch operation.\n")
      .def("submit_batch_exists", &Hf3fsConnector::submit_batch_exists,
           py::arg("keys"),
           "Submit a batch EXISTS operation.\n\n"
           "Args:\n"
           "    keys: list of key strings to check.\n\n"
           "Returns:\n"
           "    int: future id for tracking this batch operation.\n")
      .def("submit_batch_delete", &Hf3fsConnector::submit_batch_delete,
           py::arg("keys"),
           "Submit a batch DELETE operation.\n\n"
           "Args:\n"
           "    keys: list of key strings to delete.\n\n"
           "Returns:\n"
           "    int: future id for tracking this batch operation.\n")
      .def_static(
          "parse_key",
          [](const std::string& key) {
            // Parse key format "high:low" or "chunk_id" directly
            auto c = key.find(':');
            if (c == std::string::npos) {
              try {
                return static_cast<uint64_t>(std::stoull(key));
              } catch (...) {
              }
            } else {
              try {
                uint64_t h =
                    static_cast<uint64_t>(std::stoull(key.substr(0, c)));
                uint64_t l =
                    static_cast<uint64_t>(std::stoull(key.substr(c + 1)));
                return static_cast<uint64_t>((h << 32) | l);
              } catch (...) {
              }
            }
            throw std::runtime_error("Failed to parse key: " + key);
          },
          py::arg("key"),
          "Parse a key string and extract the chunk_id.\n\n"
          "Args:\n"
          "    key: The key string (e.g., '0:12345' or '12345').\n\n"
          "Returns:\n"
          "    int: The parsed chunk_id.\n")
      .def_static(
          "key_to_path",
          [](const std::string& key) {
            std::string result;
            result.reserve(key.size());
            for (char c : key) {
              switch (c) {
                case ':':
                case '@':
                case '/':
                case '\\':
                  result += '_';
                  break;
                default:
                  result += c;
                  break;
              }
            }
            return "/lmcache/" + result;
          },
          py::arg("key"),
          "Convert a key string to a 3FS path.\n\n"
          "Args:\n"
          "    key: The key string.\n\n"
          "Returns:\n"
          "    str: The 3FS path (e.g., 'lmcache/0_12345').\n");
}
