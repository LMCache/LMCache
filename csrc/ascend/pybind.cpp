// SPDX-License-Identifier: Apache-2.0
//
// Host-side bindings for the Ascend NPU MP native surface.  Only the symbols
// the in-tree MP runtime uses are exported; SP (in-process) kernels stay in
// the LMCache-Ascend plugin.

#include "completion_recorder.h"
#include "event_recorder.h"
#include "mem_alloc.h"
#include "mp_mem_kernels.h"
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <torch/torch.h>
#include <vector>

namespace py = pybind11;

namespace {
constexpr size_t kMaxPlanes = 4;

py::tuple plane_array_get(const int64_t* arr, int32_t n) {
  if (n <= 0) {
    return py::tuple();
  }
  const size_t len =
      static_cast<size_t>(n) < kMaxPlanes ? static_cast<size_t>(n) : kMaxPlanes;
  py::tuple out(len);
  for (size_t i = 0; i < len; ++i) {
    out[i] = arr[i];
  }
  return out;
}

void plane_array_set(int64_t* arr, const std::vector<int64_t>& v,
                     const char* name) {
  if (v.size() > kMaxPlanes) {
    throw py::value_error(std::string(name) + " length exceeds max 4");
  }
  for (size_t i = 0; i < kMaxPlanes; ++i) {
    arr[i] = i < v.size() ? v[i] : 0;
  }
}
}  // namespace

PYBIND11_MODULE(npu_ops, m) {
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);

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

  py::enum_<TransferDirection>(m, "TransferDirection", py::module_local())
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  py::enum_<EngineKVFormat>(m, "EngineKVFormat", py::module_local())
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
      .value("NL_X_NB_BS_NH_TWO_HS", EngineKVFormat::NL_X_NB_BS_NH_TWO_HS)
      .value("NL_X_NB_NH_BS_CS", EngineKVFormat::NL_X_NB_NH_BS_CS)
      .value("NL_X_NB_BS_NH_CS", EngineKVFormat::NL_X_NB_BS_NH_CS)
      .value("NL_X_NB_BSV_BSS", EngineKVFormat::NL_X_NB_BSV_BSS)
      .value("NL_X_TWO_NB_NH_ONE_BS_HS",
             EngineKVFormat::NL_X_TWO_NB_NH_ONE_BS_HS)
      .value("NL_X_TWO_X_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS)
      .value("NL_X_NP_X_NB_BS_ONE_HS", EngineKVFormat::NL_X_NP_X_NB_BS_ONE_HS)
      .value("NL_X_NB_BS_NH_HS", EngineKVFormat::NL_X_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_NH_ONE_BS_HS",
             EngineKVFormat::NL_X_NB_TWO_NH_ONE_BS_HS)
      .export_values();
  // Match upstream lmcache_native / CUDA pybind: tests and connectors use
  // GPUKVFormat as an alias of EngineKVFormat.
  m.attr("GPUKVFormat") = m.attr("EngineKVFormat");

  // Factory constructs this class via device_ops.PageBufferShapeDesc()
  // after NpuDeviceOps.bind_native. Kernels take it by value (same as CUDA
  // + lmcache_native). dynamic_attr keeps torch dtype for the Python
  // torch_ops fallback.
  py::class_<PageBufferShapeDesc>(m, "PageBufferShapeDesc", py::module_local(),
                                  py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("kv_size", &PageBufferShapeDesc::kv_size)
      .def_readwrite("nl", &PageBufferShapeDesc::nl)
      .def_readwrite("nb", &PageBufferShapeDesc::nb)
      .def_readwrite("bs", &PageBufferShapeDesc::bs)
      .def_readwrite("nh", &PageBufferShapeDesc::nh)
      .def_readwrite("hs", &PageBufferShapeDesc::hs)
      .def_readwrite("element_size", &PageBufferShapeDesc::element_size)
      .def_readwrite("block_stride_elems",
                     &PageBufferShapeDesc::block_stride_elems)
      .def_readwrite("num_planes", &PageBufferShapeDesc::num_planes)
      .def_property(
          "plane_slot_bytes",
          [](const PageBufferShapeDesc& s) {
            return plane_array_get(s.plane_slot_bytes, s.num_planes);
          },
          [](PageBufferShapeDesc& s, const std::vector<int64_t>& v) {
            plane_array_set(s.plane_slot_bytes, v, "plane_slot_bytes");
          })
      .def_property(
          "plane_block_stride_bytes",
          [](const PageBufferShapeDesc& s) {
            return plane_array_get(s.plane_block_stride_bytes, s.num_planes);
          },
          [](PageBufferShapeDesc& s, const std::vector<int64_t>& v) {
            plane_array_set(s.plane_block_stride_bytes, v,
                            "plane_block_stride_bytes");
          });

  m.def(
      "multi_layer_block_kv_transfer",
      [](const torch::Tensor& paged_buffer_ptrs_tensor,
         std::vector<int64_t> lmcache_objects_ptrs,
         const torch::Tensor& block_ids, const torch::Device& device,
         const py::object& direction, PageBufferShapeDesc shape_desc,
         int lmcache_chunk_size, const py::object& engine_kv_format,
         int skip_prefix_n_blocks) {
        const auto dir =
            static_cast<TransferDirection>(py::int_(direction).cast<int>());
        const auto fmt =
            static_cast<EngineKVFormat>(py::int_(engine_kv_format).cast<int>());
        // Direct launch path: release the GIL while the transfer drains
        // torch_npu's ACL task queue (getCurrentNPUStream().stream()) and
        // enqueues kernels + memcpys. A c10::Error from TORCH_CHECK
        // re-acquires the GIL during unwinding, so failures still surface as
        // Python exceptions (mp worker threads keep a clean error contract).
        py::gil_scoped_release release;
        multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor, std::move(lmcache_objects_ptrs),
            block_ids, device, dir, shape_desc, lmcache_chunk_size, fmt,
            skip_prefix_n_blocks);
      },
      py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
      py::arg("block_ids"), py::arg("device"), py::arg("direction"),
      py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
      py::arg("engine_kv_format"), py::arg("skip_prefix_n_blocks"));

  py::class_<StagingCopy>(m, "StagingCopy", py::module_local())
      .def(py::init([](uintptr_t dest, uintptr_t src, size_t nbytes,
                       size_t host_offset) {
             return StagingCopy{dest, src, nbytes, host_offset};
           }),
           py::arg("dest"), py::arg("src"), py::arg("nbytes"),
           py::arg("host_offset"));
  py::class_<LaunchVar>(m, "LaunchVar", py::module_local())
      .def(
          py::init([](int group_idx, int64_t block_ids_offset, int total_blocks,
                      int num_objects, int skip_prefix_n_blocks) {
            return LaunchVar{group_idx, block_ids_offset, total_blocks,
                             num_objects, skip_prefix_n_blocks};
          }),
          py::arg("group_idx"), py::arg("block_ids_offset"),
          py::arg("total_blocks"), py::arg("num_objects"),
          py::arg("skip_prefix_n_blocks"));
  py::class_<BatchStep>(m, "BatchStep", py::module_local())
      .def(py::init([](std::vector<StagingCopy> staging,
                       std::vector<LaunchVar> launches) {
             return BatchStep{std::move(staging), std::move(launches)};
           }),
           py::arg("staging"), py::arg("launches"));
  py::class_<KernelGroupSpec>(m, "KernelGroupSpec", py::module_local())
      .def(py::init([](uintptr_t paged_buffer_ptrs,
                       std::vector<int64_t> lmcache_objects_ptrs,
                       PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
                       int engine_kv_format, uintptr_t block_ids_base,
                       int64_t block_ids_capacity) {
             return KernelGroupSpec{
                 paged_buffer_ptrs,
                 std::move(lmcache_objects_ptrs),
                 std::move(shape_desc),
                 lmcache_chunk_size,
                 static_cast<EngineKVFormat>(engine_kv_format),
                 block_ids_base,
                 block_ids_capacity};
           }),
           py::arg("paged_buffer_ptrs"), py::arg("lmcache_objects_ptrs"),
           py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
           py::arg("engine_kv_format"), py::arg("block_ids_base"),
           py::arg("block_ids_capacity"));

  m.def(
      "execute_object_group_transfer",
      [](int direction, const torch::Device& device,
         size_t host_buffer_alignment,
         const std::vector<KernelGroupSpec>& kernel_group_specs,
         const std::vector<BatchStep>& batch_steps) {
        return execute_object_group_transfer(
            static_cast<TransferDirection>(direction), device,
            host_buffer_alignment, kernel_group_specs, batch_steps);
      },
      py::arg("direction"), py::arg("device"), py::arg("host_buffer_alignment"),
      py::arg("kernel_group_specs"), py::arg("batch_steps"),
      // Direct launch path (CUDA parity): every kernel launch and memcpy is
      // enqueued on the calling thread after draining torch_npu's ACL task
      // queue (getCurrentNPUStream().stream()), so the GIL can be released
      // for the whole enqueue.
      py::call_guard<py::gil_scoped_release>());
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::call_guard<py::gil_scoped_release>());
}
