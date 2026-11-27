// SPDX-FileCopyrightText: (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// AVX-512 build of the GHA CPU bindings (lmcache.gha_quant_avx512).
// A separate translation unit so its object file does not collide with the
// AVX2 build of the same source; compiled with the AVX-512 flags and
// GHA_QUANT_MODULE_NAME=gha_quant_avx512 (see build_profiles/sycl.py).
#include "../gha_quant_wrapper.cpp"
