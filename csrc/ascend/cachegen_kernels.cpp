#include "cachegen_kernels.h"
#include <pybind11/pybind11.h>
#include <Python.h> 

namespace py = pybind11;

void encode_cuda_new(const at::Tensor& cdf, const at::Tensor& input_sym,
                     at::Tensor& output_buffer, at::Tensor& output_lengths) {
    // TODO:
    PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
    throw py::error_already_set();
};

void decode_cuda_new(const at::Tensor& cdf, const at::Tensor& bytestreams,
                     const at::Tensor& lengths, at::Tensor& output) {
    // TODO:
    PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
    throw py::error_already_set();
};

void decode_cuda_prefsum(const at::Tensor& cdf, const at::Tensor& bytestreams,
                         const at::Tensor& lengths, at::Tensor& output) {
    // TODO:
    PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
    throw py::error_already_set();
};

at::Tensor calculate_cdf(const at::Tensor& input, const int max_bins) {
    // TODO:
    PyErr_SetString(PyExc_NotImplementedError, "Please contact LMCache Ascend.");
    throw py::error_already_set();
};