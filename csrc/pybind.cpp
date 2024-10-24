#include <pybind11/pybind11.h>
#include "mem_kernels.cuh"
#include "cachegen_kernels.cuh"
#include <torch/torch.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(lmc_ops, m) {
    m.def("load_and_reshape_flash", &load_and_reshape_flash, "A function that loads the kv cache from paged memory");
    m.def("encode_fast_new", &encode_cuda_new);
    m.def("decode_fast_new", &decode_cuda_new);
    m.def("decode_fast_prefsum", &decode_cuda_prefsum);
    m.def("calculate_cdf", &calculate_cdf);
}