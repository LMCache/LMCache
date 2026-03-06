// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include "mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include "pos_kernels.cuh"
#include "mem_alloc.h"
#include "utils.h"
#include <torch/torch.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
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
      .export_values();
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"),
        py::arg("gpu_kv_format"), py::arg("block_size") = 0,
        py::arg("skip_prefix_n_tokens") = 0);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer,
        py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
        py::arg("slot_mapping"), py::arg("direction"), py::arg("gpu_kv_format"),
        py::arg("token_major") = false);
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl,
        py::arg("lmc_key_value_cache"), py::arg("sgl_key_cache"),
        py::arg("sgl_value_cache"), py::arg("slot_mapping"),
        py::arg("direction"), py::arg("token_major") = false);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async);
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("alloc_numa_ptr", &alloc_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_numa_ptr", &free_numa_ptr);
  m.def("alloc_shm_pinned_ptr", &alloc_shm_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_shm_pinned_ptr", &free_shm_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);
}