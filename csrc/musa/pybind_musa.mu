// SPDX-License-Identifier: Apache-2.0
// pybind surface for the MUSA codec kernels (mirrors csrc/cuda/pybind.cpp).
//
// Only the CacheGen arithmetic-codec entry points are exported; the Python
// names match ``lmcache.cuda_ops`` so ``MusaDeviceOps`` can dispatch to them
// directly from its codec overrides.
//
// Compiled as ``.mu`` so mcc picks it up alongside the kernel sources; the
// content is pure host code.

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "cachegen_kernels_musa.h"

namespace py = pybind11;

PYBIND11_MODULE(musa_ops, m) {
  m.def("encode_fast_new", &encode_musa_new);
  m.def("decode_fast_new", &decode_musa_new);
  m.def("decode_fast_prefsum", &decode_musa_prefsum);
  m.def("calculate_cdf", &calculate_cdf_musa);
}
