// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// AVX-512 build of the GHA CPU kernels (lmcache.gha_quant_avx512).
// A separate translation unit so its object file does not collide with the
// AVX2 build of the same source (see build_profiles/sycl.py).
#include "../quant.cpp"
