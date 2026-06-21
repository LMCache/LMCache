// SPDX-License-Identifier: Apache-2.0

//
// Python bindings for the SYCL/XPU memory and CacheGen kernels.
// Exposed as `lmcache.xpu_ops`.
//
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include "mem_kernels_sycl.h"
#include "cachegen_kernels_sycl.h"

namespace py = pybind11;

PYBIND11_MODULE(xpu_ops, m) {
  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  // Members are single-sourced in engine_kv_format.def. The X macro's second
  // parameter must NOT be named `value`: it would be substituted into the
  // `.value(` call in the body. The numeric value is unused in this context.
  auto engine_kv_format = py::enum_<EngineKVFormat>(m, "EngineKVFormat");
#define X(name, val) engine_kv_format.value(#name, EngineKVFormat::name);
#include "../engine_kv_format.def"
#undef X
  engine_kv_format.export_values();
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"),
        py::arg("engine_kv_format"), py::arg("block_size") = 0,
        py::arg("head_size") = 0, py::arg("skip_prefix_n_tokens") = 0,
        py::call_guard<py::gil_scoped_release>());
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer,
        py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
        py::arg("slot_mapping"), py::arg("direction"),
        py::arg("engine_kv_format"), py::arg("token_major") = false,
        py::call_guard<py::gil_scoped_release>());
  m.def("single_layer_kv_transfer_sgl", &single_layer_kv_transfer_sgl,
        py::arg("lmc_key_value_cache"), py::arg("sgl_key_cache"),
        py::arg("sgl_value_cache"), py::arg("slot_mapping"),
        py::arg("direction"), py::arg("token_major") = false,
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral, py::arg("key_value"),
        py::arg("key_value_ptrs"), py::arg("slot_mapping"),
        py::arg("paged_memory_device"), py::arg("page_buffer_size"),
        py::arg("direction"), py::arg("engine_kv_format"),
        py::call_guard<py::gil_scoped_release>());
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::call_guard<py::gil_scoped_release>());

  // CacheGen / RoPE kernels (Intel XPU).  Names match the
  // lmcache.python_ops_fallback module so lmcache._get_backend() can
  // transparently override.
  m.def("calculate_cdf", &calculate_cdf_xpu, py::arg("input"),
        py::arg("max_bins"));
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused_xpu,
        py::arg("old_positions"), py::arg("new_positions"), py::arg("key"),
        py::arg("head_size"), py::arg("cos_sin_cache"), py::arg("is_neox"));
  m.def("encode_fast_new", &encode_fast_new_xpu, py::arg("cdf"),
        py::arg("input_sym"), py::arg("output_buffer"),
        py::arg("output_lengths"));
  m.def("decode_fast_new", &decode_fast_new_xpu, py::arg("cdf"),
        py::arg("bytestreams"), py::arg("lengths"), py::arg("output"));
  m.def("decode_fast_prefsum", &decode_fast_prefsum_xpu, py::arg("cdf"),
        py::arg("bytestreams"), py::arg("lengths_prefsum"), py::arg("output"));
  // Backward-compat alias: GPUKVFormat -> EngineKVFormat
  m.attr("GPUKVFormat") = m.attr("EngineKVFormat");
}
