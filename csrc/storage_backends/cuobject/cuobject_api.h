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
#ifdef CU_OBJ_SUCCESS
#undef CU_OBJ_SUCCESS
#endif
constexpr int CU_OBJ_SUCCESS = 0;

#ifdef CU_OBJ_FAIL
#undef CU_OBJ_FAIL
#endif
constexpr int CU_OBJ_FAIL = 1;

#ifdef CUOBJ_PROTO_RDMA_DC_V1
#undef CUOBJ_PROTO_RDMA_DC_V1
#endif
constexpr int CUOBJ_PROTO_RDMA_DC_V1 = 1001;

#ifdef CUOBJ_MAX_MEMORY_REG_SIZE
#undef CUOBJ_MAX_MEMORY_REG_SIZE
#endif
constexpr size_t CUOBJ_MAX_MEMORY_REG_SIZE = 4ULL * 1024 * 1024 * 1024;

// Operation types for cuMemObjGetRDMAToken().
#ifdef CUOBJ_OP_GET
#undef CUOBJ_OP_GET
#endif
constexpr int CUOBJ_OP_GET = 0;  // CUOBJ_GET

#ifdef CUOBJ_OP_PUT
#undef CUOBJ_OP_PUT
#endif
constexpr int CUOBJ_OP_PUT = 1;  // CUOBJ_PUT

#ifdef CUOBJ_MEMORY_SYSTEM
#undef CUOBJ_MEMORY_SYSTEM
#endif
constexpr int CUOBJ_MEMORY_SYSTEM = 0;

#ifdef CUOBJ_MEMORY_CUDA_MANAGED
#undef CUOBJ_MEMORY_CUDA_MANAGED
#endif
constexpr int CUOBJ_MEMORY_CUDA_MANAGED = 1;

#ifdef CUOBJ_MEMORY_CUDA_DEVICE
#undef CUOBJ_MEMORY_CUDA_DEVICE
#endif
constexpr int CUOBJ_MEMORY_CUDA_DEVICE = 2;

}  // namespace cuobject
}  // namespace lmcache
