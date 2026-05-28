// SPDX-License-Identifier: Apache-2.0

//
// Python bindings for the SYCL/XPU memory kernels.
// This module (lmcache.xpu_ops) mirrors the mem-kernel subset of
// lmcache.xpu_ops but targets Intel XPU via SYCL.
//
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "mem_kernels_sycl.h"

namespace py = pybind11;

PYBIND11_MODULE(xpu_ops, m) {
  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  py::enum_<GPUKVFormat>(m, "GPUKVFormat")
      .value("NB_NL_TWO_BS_NH_HS", GPUKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", GPUKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", GPUKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", GPUKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", GPUKVFormat::NL_X_NBBS_ONE_HS)
      .value("NL_X_TWO_NB_NH_BS_HS", GPUKVFormat::NL_X_TWO_NB_NH_BS_HS)
      .value("NL_X_NB_TWO_NH_BS_HS", GPUKVFormat::NL_X_NB_TWO_NH_BS_HS)
      .value("NB_NL_TWO_NH_BS_HS", GPUKVFormat::NB_NL_TWO_NH_BS_HS)
      .export_values();
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"),
        py::arg("gpu_kv_format"), py::arg("block_size") = 0,
        py::arg("skip_prefix_n_tokens") = 0,
        py::call_guard<py::gil_scoped_release>());
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer,
        py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
        py::arg("slot_mapping"), py::arg("direction"), py::arg("gpu_kv_format"),
        py::arg("token_major") = false,
        py::call_guard<py::gil_scoped_release>());
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::call_guard<py::gil_scoped_release>());
}
