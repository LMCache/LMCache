// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/cuda_metadata.h"

namespace lmcache::mp {

std::optional<std::uint64_t> DtypeElementSize(const std::string& dtype) {
  if (dtype == "torch.float16" || dtype == "torch.bfloat16" ||
      dtype == "torch.half") {
    return 2;
  }
  if (dtype == "torch.float32" || dtype == "torch.float") {
    return 4;
  }
  if (dtype == "torch.float64" || dtype == "torch.double") {
    return 8;
  }
  if (dtype == "torch.uint8" || dtype == "torch.int8" ||
      dtype == "torch.bool" || dtype.find("float8") != std::string::npos) {
    return 1;
  }
  if (dtype == "torch.int16" || dtype == "torch.short") {
    return 2;
  }
  if (dtype == "torch.int32" || dtype == "torch.int") {
    return 4;
  }
  if (dtype == "torch.int64" || dtype == "torch.long") {
    return 8;
  }
  return std::nullopt;
}

std::string IpcHandleCacheKey(const KvTensorMetadata& metadata) {
  std::string key = metadata.device_uuid;
  key.push_back('\0');
  key.append(reinterpret_cast<const char*>(metadata.ipc_handle.data()),
             metadata.ipc_handle.size());
  return key;
}

}  // namespace lmcache::mp

