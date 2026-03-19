// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "multi_layer_block_kv_transfer.cuh"

namespace py = pybind11;

PYBIND11_MODULE(kernel_harness_ops, m) {
  m.doc() = "LMCache kernel harness: block-level multi-layer KV transfer";

  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();

  py::enum_<GPUKVFormat>(m, "GPUKVFormat")
      .value("NB_NL_TWO_BS_NH_HS", GPUKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", GPUKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_BS_HS", GPUKVFormat::NL_X_NB_BS_HS)
      .export_values();

  m.def("multi_layer_block_kv_transfer", &multi_layer_block_kv_transfer,
        py::arg("key_value_tensors"), py::arg("memory_objects"),
        py::arg("block_ids"), py::arg("device"), py::arg("direction"),
        py::arg("gpu_kv_format"), py::arg("block_size"), py::arg("num_blocks"),
        py::arg("skip_prefix_n_blocks"),
        py::call_guard<py::gil_scoped_release>());
}
