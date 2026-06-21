// SPDX-License-Identifier: Apache-2.0

#pragma once

// Backend-agnostic descriptors for KV-cache transfers.
//
// These enums describe *what* is being transferred (the direction of a
// host/device copy and the physical KV-cache memory layout) and carry no
// accelerator-specific dependency. They are intentionally free of any
// <torch/...>, <ATen/...> or vendor runtime headers so that every backend
// (CUDA, ROCm, MUSA, SYCL/XPU, ...) can share a single definition instead of
// redeclaring them in each accelerator-coupled kernel header.

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

/*
Symbol Reference:
NL: number of layers
NB: number of blocks/pages
BS: block/page size
NBBS: block/page buffer size = NB * BS
NH: number of heads
HS: head size
TWO: 2
ONE: 1

_ means a dimension within the same tensor
_X_ means a dimension across a list

A_X_B_X_C_D_E means:
kv_cache: List[List[torch.Tensor]]
len(kv_cache) = A
len(kv_cache[0]) = B
kv_cache[0][0].shape = (C, D, E)

The logic for identifying the format currently lives in
`lmcache/v1/gpu_connector/utils.py`
*/
// Members are single-sourced in engine_kv_format.def (see that file for the
// per-format documentation). Each X(name, value) becomes an enum member.
enum class EngineKVFormat : int {
#define X(name, value) name = value,
#include "engine_kv_format.def"
#undef X
};
