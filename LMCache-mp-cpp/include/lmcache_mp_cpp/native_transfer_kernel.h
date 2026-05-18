#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace lmcache::mp {

struct NativePageBufferShapeDesc {
  int kv_size = 0;
  int nl = 0;
  int nb = 0;
  int bs = 0;
  int nh = 0;
  int hs = 0;
  int element_size = 0;
  int block_stride_elems = 0;
};

bool NativeCudaBlockTransfer(const std::vector<void*>& paged_buffer_ptrs,
                             const std::vector<void*>& lmcache_object_ptrs,
                             const std::vector<std::int64_t>& block_ids,
                             bool lmcache_to_engine,
                             NativePageBufferShapeDesc shape_desc,
                             int lmcache_chunk_size, int gpu_kv_format,
                             int skip_prefix_n_blocks, std::string* error);

bool NativeCudaBlockTransferWithStream(
    const std::vector<void*>& paged_buffer_ptrs,
    const std::vector<void*>& lmcache_object_ptrs,
    const std::vector<std::int64_t>& block_ids, bool lmcache_to_engine,
    NativePageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    int gpu_kv_format, int skip_prefix_n_blocks, void* cuda_stream,
    bool synchronize, std::string* error);

}  // namespace lmcache::mp
