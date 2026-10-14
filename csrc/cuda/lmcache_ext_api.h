// SPDX-License-Identifier: Apache-2.0
//
// Function table that lmcache.cuda_ops exports to separately built CUDA
// extensions (plugin transfer modules).
//
// Kernels and launchers are header-only (mp_mem_kernels_detail.cuh), but the
// object-group plan executor calls code compiled into cuda_ops (staging
// copies, phase timing), and Python loads extensions with RTLD_LOCAL, so an
// extension cannot link against it. cuda_ops instead publishes this table as
// the PyCapsule `lmcache.cuda_ops._ext_api`; an extension fetches it once with
// lmcache_ext::import_api() and calls through it.
//
// Compatibility: `version` changes on any incompatible change, including a
// layout change of a type in a signature below; entries are only ever
// appended, and `size` tells how many the running cuda_ops has.

#pragma once

#include <Python.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "kv_transfer_plan_types.h"  // KernelGroupSpec, TransferDirection
#include "transfer_plan_types.cuh"   // BatchStep, LaunchVar

#define LMCACHE_EXT_API_VERSION 1
#define LMCACHE_EXT_API_CAPSULE_NAME "lmcache.cuda_ops._ext_api"

namespace lmcache_ext {

// Performs one kernel launch of a plan in place of cuda_ops' own kernel.
// Called after the executor has bounds-checked `launch` against `group`, with
// the plan's device current; `block_ids` points at this launch's slice of
// `group.block_ids_base` (`launch.total_blocks` entries, device memory), and
// the launch must be enqueued on `stream`. May throw (e.g. c10::Error); the
// exception propagates out of execute_object_group_transfer.
using LaunchFn = void (*)(void* user_data, const KernelGroupSpec& group,
                          const LaunchVar& launch, const int64_t* block_ids,
                          cudaStream_t stream);

struct Api {
  uint32_t version;  // LMCACHE_EXT_API_VERSION of the running cuda_ops
  uint32_t size;     // sizeof(Api) of the running cuda_ops

  // Layout guard: sizeof each plan type as compiled into cuda_ops.
  uint32_t sizeof_page_buffer_shape_desc;
  uint32_t sizeof_kernel_group_spec;
  uint32_t sizeof_launch_var;
  uint32_t sizeof_batch_step;

  // Runs a plan exactly like cuda_ops.execute_object_group_transfer (device
  // guard, staging copies, bounds checks, ordering, phase timing), except that
  // each kernel launch is delegated to `launch_fn`. Call without the GIL.
  void (*execute_object_group_transfer)(
      TransferDirection direction, int device_index,
      size_t host_buffer_alignment,
      const std::vector<KernelGroupSpec>& kernel_group_specs,
      const std::vector<BatchStep>& batch_steps, LaunchFn launch_fn,
      void* user_data, bool phase_timing_enabled, const char* session_id);
};

// Fetches the table from lmcache.cuda_ops (importing it if needed) and checks
// it matches the headers this extension was built with. Call with the GIL
// held, e.g. from PYBIND11_MODULE. On failure returns nullptr with a Python
// exception set.
inline const Api* import_api() {
  const auto* api = static_cast<const Api*>(
      PyCapsule_Import(LMCACHE_EXT_API_CAPSULE_NAME, 0));
  if (api == nullptr) {
    return nullptr;
  }
  if (api->version != LMCACHE_EXT_API_VERSION || api->size < sizeof(Api) ||
      api->sizeof_page_buffer_shape_desc != sizeof(PageBufferShapeDesc) ||
      api->sizeof_kernel_group_spec != sizeof(KernelGroupSpec) ||
      api->sizeof_launch_var != sizeof(LaunchVar) ||
      api->sizeof_batch_step != sizeof(BatchStep)) {
    PyErr_Format(PyExc_ImportError,
                 "%s (version %u) is incompatible with the lmcache headers "
                 "this extension was built with (version %d); rebuild it "
                 "against the installed lmcache",
                 LMCACHE_EXT_API_CAPSULE_NAME, api->version,
                 LMCACHE_EXT_API_VERSION);
    return nullptr;
  }
  return api;
}

}  // namespace lmcache_ext
