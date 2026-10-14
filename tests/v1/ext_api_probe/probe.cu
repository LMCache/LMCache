// SPDX-License-Identifier: Apache-2.0
//
// A minimal extension built only against lmcache.get_include(),
// the way an external transfer module is. It runs object-group plans through
// the lmcache.cuda_ops._ext_api executor, launching LMCache's kernel itself
// with the object pointers read from a device array and, optionally, a custom
// object-offset policy. Used by tests/v1/test_ext_api.py.

#include <torch/extension.h>

#include "cuda/lmcache_ext_api.h"
#include "cuda/mp_mem_kernels_detail.cuh"

namespace {

const lmcache_ext::Api* g_api = nullptr;

// Stores layer l of the plan at layer (nl - 1 - l) of the LMCache object.
struct ReversedLayerOffset {
  template <typename ScalarType, EngineKVFormat format>
  __device__ size_t offset(int k_or_v, int token_offset, int layer_idx,
                           int lmcache_chunk_size,
                           const PageBufferShapeDesc& shape_desc) const {
    return lmcache_mp::TwoLTDObjectOffset{}.offset<ScalarType, format>(
        k_or_v, token_offset, shape_desc.nl - 1 - layer_idx, lmcache_chunk_size,
        shape_desc);
  }
};

struct ProbeLaunch {
  TransferDirection direction;
  const int64_t* object_ptrs;  // device array, one entry per object slot
  bool reverse_layers;
};

void launch(void* user_data, const KernelGroupSpec& group,
            const LaunchVar& launch, const int64_t* block_ids,
            cudaStream_t stream) {
  const auto& probe = *static_cast<const ProbeLaunch*>(user_data);
  lmcache_mp::dispatch_by_transfer_unit(
      group.shape_desc, group.engine_kv_format, [&](auto unit) {
        using T = typename decltype(unit)::type;
        const auto objects = reinterpret_cast<T* const*>(probe.object_ptrs);
        const auto paged = reinterpret_cast<T**>(group.paged_buffer_ptrs);
        const auto run = [&](auto offset) {
          lmcache_mp::launch_multi_layer_block_transfer<T>(
              objects, launch.num_objects, paged, block_ids,
              launch.total_blocks, probe.direction, group.shape_desc,
              group.lmcache_chunk_size, group.engine_kv_format,
              launch.skip_prefix_n_blocks, stream, offset);
        };
        if (probe.reverse_layers) {
          run(ReversedLayerOffset{});
        } else {
          run(lmcache_mp::TwoLTDObjectOffset{});
        }
      });
}

void execute(int direction, int device_index, size_t host_buffer_alignment,
             const std::vector<KernelGroupSpec>& kernel_group_specs,
             const std::vector<BatchStep>& batch_steps,
             const torch::Tensor& object_ptrs, bool reverse_layers) {
  TORCH_CHECK(object_ptrs.is_cuda() && object_ptrs.dtype() == torch::kInt64,
              "object_ptrs must be a CUDA int64 tensor");
  ProbeLaunch probe{static_cast<TransferDirection>(direction),
                    object_ptrs.data_ptr<int64_t>(), reverse_layers};
  g_api->execute_object_group_transfer(
      probe.direction, device_index, host_buffer_alignment, kernel_group_specs,
      batch_steps, &launch, &probe, /*phase_timing_enabled=*/false, "");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  g_api = lmcache_ext::import_api();
  if (g_api == nullptr) {
    throw py::error_already_set();
  }
  m.def("execute", &execute, py::arg("direction"), py::arg("device_index"),
        py::arg("host_buffer_alignment"), py::arg("kernel_group_specs"),
        py::arg("batch_steps"), py::arg("object_ptrs"),
        py::arg("reverse_layers") = false,
        py::call_guard<py::gil_scoped_release>());
}
