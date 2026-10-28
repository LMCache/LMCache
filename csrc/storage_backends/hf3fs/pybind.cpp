// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText:Copyright (c) 2026 Samsung Electronics Co., Ltd.

#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include "../connector_pybind_utils.h"
#include "connector.h"

namespace py = pybind11;

// Decode a wire key into ObjectKey fields.
//
// Wire key format (from ``_object_key_to_string`` in
// native_connector_l2_adapter.py):
//   <model>@<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>]
// Unlike the on-disk filename, kv_rank carries no "0x" prefix and
// model_name already contains its original '/' separators.
//
// Returns false (leaving ``out`` untouched) when the wire key does not
// match this shape or any hex field fails to decode.
bool parse_wire_key(const std::string& wire_key, std::string& model_name,
                    uint32_t& kv_rank, uint32_t& object_group_id,
                    std::string& cache_salt, std::string& chunk_hash_hex) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= wire_key.size(); ++pos) {
    if (pos == wire_key.size() || wire_key[pos] == '@') {
      parts.emplace_back(wire_key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 4 && parts.size() != 5) {
    return false;
  }

  auto hex_value = [](const std::string& text, uint32_t& value) -> bool {
    if (text.empty()) {
      return false;
    }
    uint32_t result = 0;
    for (char c : text) {
      uint32_t digit;
      if (c >= '0' && c <= '9') {
        digit = static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        digit = static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        digit = static_cast<uint32_t>(c - 'A' + 10);
      } else {
        return false;
      }
      result = result * 16 + digit;
    }
    value = result;
    return true;
  };

  const std::string& hash = parts[3];
  if (hash.empty() || hash.size() % 2 != 0) {
    return false;
  }
  uint32_t rank = 0;
  uint32_t ogid = 0;
  if (!hex_value(parts[1], rank) || !hex_value(parts[2], ogid)) {
    return false;
  }

  model_name = parts[0];
  kv_rank = rank;
  object_group_id = ogid;
  cache_salt = parts.size() == 5 ? parts[4] : std::string();
  chunk_hash_hex = hash;
  return true;
}

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
                       bool enable_key_buffer, bool collect_file_sizes,
                       py::object per_op_workers) {
             return new lmcache::connector::Hf3fsConnector(
                 std::move(mount_point), std::move(base_paths), num_workers,
                 ior_entries, io_depth, numa_id, iov_size, time_out,
                 enable_key_buffer, collect_file_sizes,
                 lmcache::connector::pybind_utils::parse_per_op_workers(
                     per_op_workers));
           }),
           py::arg("mount_point"), py::arg("base_paths"),
           py::arg("num_workers") = 8, py::arg("ior_entries") = 256,
           py::arg("io_depth") = 0, py::arg("numa_id") = -1,
           py::arg("iov_size") = 209715200, py::arg("time_out") = 200,
           py::arg("enable_key_buffer") = true,
           py::arg("collect_file_sizes") = false,
           py::arg("per_op_workers") = py::none())
          LMCACHE_BIND_CONNECTOR_METHODS(lmcache::connector::Hf3fsConnector)

      // Return (ObjectKey, file_size) from the client's startup scan.
      // Returns an empty list when collect_file_sizes was disabled
      //  (no capacity accounting).
      .def("get_existing_keys_with_sizes",
           [](lmcache::connector::Hf3fsConnector& self) {
             // Lazily resolve the callable Python ObjectKey so pybind never
             // depends on import order between this module and the
             // ``lmcache.v1.distributed.api`` package.
             py::object obj_type =
                 py::module_::import("lmcache.v1.distributed.api")
                     .attr("ObjectKey");
             py::object fromhex =
                 py::module_::import("builtins").attr("bytes").attr("fromhex");

             py::list result;
             for (const auto& [wire_key, size] :
                  self.existing_keys_with_sizes()) {
               std::string model_name;
               uint32_t kv_rank = 0;
               uint32_t object_group_id = 0;
               std::string cache_salt;
               std::string chunk_hash_hex;
               try {
                 if (!parse_wire_key(wire_key, model_name, kv_rank,
                                     object_group_id, cache_salt,
                                     chunk_hash_hex)) {
                   continue;
                 }
                 result.append(py::make_tuple(
                     obj_type(py::arg("chunk_hash") = fromhex(chunk_hash_hex),
                              py::arg("model_name") = model_name,
                              py::arg("kv_rank") = kv_rank,
                              py::arg("object_group_id") = object_group_id,
                              py::arg("cache_salt") = cache_salt),
                     size));
               } catch (const std::exception& e) {
                 // A single bad key must not abort the whole hydration scan ,
                 // so skip it and keep decoding the rest
                 fprintf(stderr,
                         "[LMCache HF3FS] hydration: skipping unparsable key "
                         "%s (size %llu): %s\n",
                         wire_key.c_str(),
                         static_cast<unsigned long long>(size), e.what());
                 continue;
               }
             }
             return result;
           });
}
