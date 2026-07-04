// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mem_kernels.cuh"
#include "mp_mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include "pos_kernels.cuh"
#include "mem_alloc.h"
#include "utils.h"
#include "event_recorder.h"
#include "completion_recorder.h"
#include <torch/torch.h>
#include <torch/extension.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  py::enum_<EngineKVFormat>(m, "EngineKVFormat")
      .value("NB_NL_TWO_BS_NH_HS", EngineKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", EngineKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", EngineKVFormat::NL_X_NBBS_ONE_HS)
      .value("NL_X_TWO_NB_NH_BS_HS", EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
      .value("NL_X_NB_TWO_NH_BS_HS", EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
      .value("NB_NL_TWO_NH_BS_HS", EngineKVFormat::NB_NL_TWO_NH_BS_HS)
      .value("TWO_X_NL_X_NB_BS_NH_HS", EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS)
      .value("NL_X_NB_NH_BS_TWO_HS", EngineKVFormat::NL_X_NB_NH_BS_TWO_HS)
      .export_values();
  // Format classification, shared with the device kernels (engine_kv_format.h).
  m.def(
      "is_cross_layer", [](EngineKVFormat f) { return is_cross_layer(f); },
      py::arg("engine_kv_format"));
  m.def(
      "is_kv_list", [](EngineKVFormat f) { return is_kv_list(f); },
      py::arg("engine_kv_format"));
  m.def(
      "is_layer_list", [](EngineKVFormat f) { return is_layer_list(f); },
      py::arg("engine_kv_format"));
  m.def(
      "is_mla", [](EngineKVFormat f) { return is_mla(f); },
      py::arg("engine_kv_format"));
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"),
        py::arg("engine_kv_format"), py::arg("block_size") = 0,
        py::arg("head_size") = 0, py::arg("skip_prefix_n_tokens") = 0,
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer,
        py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
        py::arg("slot_mapping"), py::arg("direction"),
        py::arg("engine_kv_format"), py::arg("token_major") = false);
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl,
        py::arg("lmc_key_value_cache"), py::arg("sgl_key_cache"),
        py::arg("sgl_value_cache"), py::arg("slot_mapping"),
        py::arg("direction"), py::arg("token_major") = false);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::call_guard<py::gil_scoped_release>());
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_hugepage_pinned_ptr", &alloc_hugepage_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_hugepage_pinned_ptr", &free_hugepage_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("alloc_hugepage_pinned_numa_ptr", &alloc_hugepage_pinned_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_hugepage_pinned_numa_ptr", &free_hugepage_pinned_numa_ptr);
  m.def("alloc_numa_ptr", &alloc_numa_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_numa_ptr", &free_numa_ptr);
  m.def("alloc_shm_pinned_ptr", &alloc_shm_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("free_shm_pinned_ptr", &free_shm_pinned_ptr,
        py::call_guard<py::gil_scoped_release>());
  m.def("batched_memcpy", &batched_memcpy, py::arg("src_ptrs"),
        py::arg("dst_ptrs"), py::arg("sizes"),
        py::call_guard<py::gil_scoped_release>());
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);
  m.def("multi_layer_block_kv_transfer", &multi_layer_block_kv_transfer,
        py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
        py::arg("block_ids"), py::arg("device"), py::arg("direction"),
        py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
        py::arg("engine_kv_format"), py::arg("skip_prefix_n_blocks"),
        py::call_guard<py::gil_scoped_release>());
  py::class_<PageBufferShapeDesc>(m, "PageBufferShapeDesc")
      .def(py::init<>())
      .def_readwrite("kv_size", &PageBufferShapeDesc::kv_size)
      .def_readwrite("nl", &PageBufferShapeDesc::nl)
      .def_readwrite("nb", &PageBufferShapeDesc::nb)
      .def_readwrite("bs", &PageBufferShapeDesc::bs)
      .def_readwrite("nh", &PageBufferShapeDesc::nh)
      .def_readwrite("hs", &PageBufferShapeDesc::hs)
      .def_readwrite("element_size", &PageBufferShapeDesc::element_size)
      .def_readwrite("block_stride_elems",
                     &PageBufferShapeDesc::block_stride_elems);
  // Object-group transfer plan types (see mp_mem_kernels.cuh). Built on the
  // Python side and consumed by execute_object_group_transfer.
  py::class_<StagingCopy>(m, "StagingCopy")
      .def(py::init([](uintptr_t dest, uintptr_t src, size_t nbytes,
                       size_t host_offset) {
             return StagingCopy{dest, src, nbytes, host_offset};
           }),
           py::arg("dest"), py::arg("src"), py::arg("nbytes"),
           py::arg("host_offset"));
  py::class_<LaunchVar>(m, "LaunchVar")
      .def(
          py::init([](int group_idx, int64_t block_ids_offset, int total_blocks,
                      int num_objects, int skip_prefix_n_blocks) {
            return LaunchVar{group_idx, block_ids_offset, total_blocks,
                             num_objects, skip_prefix_n_blocks};
          }),
          py::arg("group_idx"), py::arg("block_ids_offset"),
          py::arg("total_blocks"), py::arg("num_objects"),
          py::arg("skip_prefix_n_blocks"));
  py::class_<BatchStep>(m, "BatchStep")
      .def(py::init([](std::vector<StagingCopy> staging,
                       std::vector<LaunchVar> launches) {
             return BatchStep{std::move(staging), std::move(launches)};
           }),
           py::arg("staging"), py::arg("launches"));
  py::class_<KernelGroupSpec>(m, "KernelGroupSpec")
      .def(py::init([](uintptr_t paged_buffer_ptrs,
                       std::vector<int64_t> lmcache_objects_ptrs,
                       PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
                       EngineKVFormat engine_kv_format,
                       uintptr_t block_ids_base, int64_t block_ids_capacity) {
             return KernelGroupSpec{
                 paged_buffer_ptrs, std::move(lmcache_objects_ptrs),
                 shape_desc,        lmcache_chunk_size,
                 engine_kv_format,  block_ids_base,
                 block_ids_capacity};
           }),
           py::arg("paged_buffer_ptrs"), py::arg("lmcache_objects_ptrs"),
           py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
           py::arg("engine_kv_format"), py::arg("block_ids_base"),
           py::arg("block_ids_capacity"));
  m.def("execute_object_group_transfer", &execute_object_group_transfer,
        py::arg("direction"), py::arg("device"),
        py::arg("host_buffer_alignment"), py::arg("kernel_group_specs"),
        py::arg("batch_steps"), py::call_guard<py::gil_scoped_release>());
  m.def("record_event_on_stream", &record_event_on_stream,
        py::arg("cuda_stream_ptr"), py::arg("event_type_name"),
        py::arg("session_id"), py::arg("str_metadata"), py::arg("int_metadata"),
        py::call_guard<py::gil_scoped_release>());
  m.def("drain_recorded_events", &drain_recorded_events);
  m.def("record_completion_on_stream", &record_completion_on_stream,
        py::arg("cuda_stream_ptr"), py::arg("kind"), py::arg("payload"),
        py::call_guard<py::gil_scoped_release>());
  // Return each payload as py::bytes; pybind11 utf-8-decodes std::string
  // by default, corrupting binary payloads (e.g. msgpack).
  m.def("drain_recorded_completions", []() {
    auto items = drain_recorded_completions();
    py::list out;
    for (auto& kv : items) {
      out.append(py::make_tuple(py::str(kv.first), py::bytes(kv.second)));
    }
    return out;
  });
  // Backward-compat alias: GPUKVFormat -> EngineKVFormat
  m.attr("GPUKVFormat") = m.attr("EngineKVFormat");
}
