#include "pos_kernels.h"
#include <pybind11/pybind11.h>
#include <Python.h> 

namespace py = pybind11;

void rotary_embedding_k_fused(const torch::Tensor& old_positions,
                              const torch::Tensor& new_positions,
                              torch::Tensor& key, int64_t head_size,
                              const torch::Tensor& cos_sin_cache, bool is_neox) {
    // TODO:
    PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
    throw py::error_already_set();
};