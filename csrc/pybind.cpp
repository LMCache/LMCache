// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include "pos_kernels.cuh"
#include "mem_alloc.h"
#include "mem_descs.h"
#include "utils.h"
#include <torch/torch.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer);
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("batched_kv_transfer", &batched_kv_transfer);
  m.def("encode_fast_new", &encode_cuda_new);
  m.def("decode_fast_new", &decode_cuda_new);
  m.def("decode_fast_prefsum", &decode_cuda_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_gpu_pci_bus_id);

  py::class_<PageBufferShapeDesc>(m, "PageBufferShapeDesc")
      .def(py::init<size_t, size_t, size_t, size_t>(), py::arg("kv_dim"),
           py::arg("num_pages"), py::arg("page_size"), py::arg("hidden_dim"),
           "Constructor for PageBufferShapeDesc")
      .def_readwrite("kv_dim", &PageBufferShapeDesc::kv_dim)
      .def_readwrite("num_pages", &PageBufferShapeDesc::num_pages)
      .def_readwrite("page_size", &PageBufferShapeDesc::page_size)
      .def_readwrite("hidden_dim", &PageBufferShapeDesc::hidden_dim)
      .def("__repr__", [](const PageBufferShapeDesc& self) {
        return "PageBufferShapeDesc(kv_dim=" + std::to_string(self.kv_dim) +
               ", num_pages=" + std::to_string(self.num_pages) +
               ", page_size=" + std::to_string(self.page_size) +
               ", hidden_dim=" + std::to_string(self.hidden_dim) + ")";
      });

  py::class_<ObjBufferShapeDesc>(m, "ObjBufferShapeDesc")
      .def(py::init<size_t, size_t, size_t, size_t>(), py::arg("kv_dim"),
           py::arg("num_layers"), py::arg("chunk_size"), py::arg("hidden_dim"),
           "Constructor for ObjBufferShapeDesc")
      .def_readwrite("kv_dim", &ObjBufferShapeDesc::kv_dim)
      .def_readwrite("num_layers", &ObjBufferShapeDesc::num_layers)
      .def_readwrite("chunk_size", &ObjBufferShapeDesc::chunk_size)
      .def_readwrite("hidden_dim", &ObjBufferShapeDesc::hidden_dim)
      .def("__repr__", [](const ObjBufferShapeDesc& self) {
        return "ObjBufferShapeDesc(kv_dim=" + std::to_string(self.kv_dim) +
               ", num_layers=" + std::to_string(self.num_layers) +
               ", chunk_size=" + std::to_string(self.chunk_size) +
               ", hidden_dim=" + std::to_string(self.hidden_dim) + ")";
      });
}
