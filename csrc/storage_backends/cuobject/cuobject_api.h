// SPDX-License-Identifier: Apache-2.0
//
// cuObject API header.
//
// Includes the NVIDIA cuObjClient header (cuobjclient.h) from the CUDA
// Toolkit (>= 13.1.1 with GPUDirect Storage / cuObject support) and
// provides namespace-scoped convenience constants for the pybind11
// module.
//
// Build prerequisites
// -------------------
//   * cuobjclient.h   in the compiler include path
//   * libcuobjclient.so   in the linker library path
//
// If the SDK is not installed, setup.py will skip building the
// lmcache_cuobject extension entirely.

#pragma once

#include <cuobjclient.h>

#include <cstddef>

namespace lmcache {
namespace cuobject {

// ---------------------------------------------------------------------------
// Re-exported constants for pybind11 convenience.
//
// Values are hardcoded integers matching the official cuObjClient API
// Specification v1.0.0 to avoid potential conflicts with vendor macros.
// ---------------------------------------------------------------------------
constexpr int CU_OBJ_SUCCESS = 0;
constexpr int CU_OBJ_FAIL = 1;

constexpr int CUOBJ_PROTO_RDMA_DC_V1 = 1001;

constexpr size_t CUOBJ_MAX_MEMORY_REG_SIZE = 4ULL * 1024 * 1024 * 1024;

// Operation types for cuMemObjGetRDMAToken().
constexpr int CUOBJ_OP_GET = 0;  // CUOBJ_GET
constexpr int CUOBJ_OP_PUT = 1;  // CUOBJ_PUT

constexpr int CUOBJ_MEMORY_SYSTEM = 0;
constexpr int CUOBJ_MEMORY_CUDA_MANAGED = 1;
constexpr int CUOBJ_MEMORY_CUDA_DEVICE = 2;

}  // namespace cuobject
}  // namespace lmcache
