#include <ATen/ATen.h>
#include <pybind11/pybind11.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>

#include <cmath>

#include <torch/extension.h>

#include <tuple>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <iterator>

#include <bitset>


const int precision = 16;
const int N = 1;
using cdf_t = uint16_t;
const int PRECISION = 16;
const int RENORMALIZATION_FACTOR = 2 << (PRECISION - 1);
const int STRIDE = 1;


std::vector<py::bytes> encode_cuda(const at::Tensor &cdf, 
                                   const at::Tensor &input_sym, 
                                   const uint32_t max_out_size,
                                   const int blockNum, 
                                   const int threadNum);

void encode_cuda_new(
        const at::Tensor &cdf, 
        const at::Tensor &input_sym,
        at::Tensor &output_buffer,
        at::Tensor &output_lengths);

void decode_cuda_new(
        const at::Tensor &cdf,
        const at::Tensor &bytestreams,
        const at::Tensor &lengths,
        at::Tensor &output);

void decode_cuda_prefsum(
        const at::Tensor &cdf,
        const at::Tensor &bytestreams,
        const at::Tensor &lengths,
        at::Tensor &output);

void decode_fast(torch::Tensor out_tensor, const at::Tensor &cdf,
                     at::Tensor concated_string, const at::Tensor start_indices, const int all_tokens, 
                     const int blockNum, const int threadNum, const int scale);

const struct cdf_ptr get_cdf_ptr_cuda(const at::Tensor &cdf);

at::Tensor calculate_cdf(
        const at::Tensor &input,
        const int max_bins);
