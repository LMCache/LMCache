// main.cpp
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <ATen/ATen.h>

extern "C" int* decode(const at::Tensor& cdf,
    std::vector<std::string>& stringList, const int all_tokens, const int blockNum, const int threadNum);

namespace py = pybind11;

PYBIND11_MODULE(torchac, m) {
    m.def("decode", &decode, "test function");
}