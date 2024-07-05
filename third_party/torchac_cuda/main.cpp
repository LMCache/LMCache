// main.cpp
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include "torchac_kernel.cuh"
//extern "C" int* decode(const at::Tensor& cdf,
//    std::vector<std::string>& stringList, const int all_tokens, const int blockNum, const int threadNum);

namespace py = pybind11;

PYBIND11_MODULE(torchac_cuda, m) {
    //m.def("decode_fast", &decode_fast);
    //m.def("encode_fast", &encode_cuda);
    m.def("encode_fast_new", &encode_cuda_new);
    m.def("decode_fast_new", &decode_cuda_new);
    m.def("decode_fast_prefsum", &decode_cuda_prefsum);
    m.def("calculate_cdf", &calculate_cdf);
}
