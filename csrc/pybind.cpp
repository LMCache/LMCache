#include <pybind11/pybind11.h>
#include "lmc_kernels.cuh"
#include <torch/torch.h>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(lmc_ops, m) {
    m.def("load_and_reshape_flash", &load_and_reshape_flash, "A function that loads the kv cache from paged memory");
}