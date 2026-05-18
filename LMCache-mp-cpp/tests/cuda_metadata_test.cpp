// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/cuda_metadata.h"

#include <cassert>
#include <string>
#include <vector>

int main() {
  using lmcache::mp::DtypeElementSize;
  using lmcache::mp::IpcHandleCacheKey;
  using lmcache::mp::KvTensorMetadata;

  assert(DtypeElementSize("torch.float16") == 2);
  assert(DtypeElementSize("torch.bfloat16") == 2);
  assert(DtypeElementSize("torch.float32") == 4);
  assert(DtypeElementSize("torch.float64") == 8);
  assert(DtypeElementSize("torch.float8_e4m3fn") == 1);
  assert(DtypeElementSize("torch.int64") == 8);
  assert(!DtypeElementSize("custom"));

  KvTensorMetadata first;
  first.device_uuid = "GPU-0";
  first.ipc_handle = {1, 2, 3};
  KvTensorMetadata second = first;
  second.ipc_handle = {1, 2, 4};
  const std::string first_key = IpcHandleCacheKey(first);
  const std::string second_key = IpcHandleCacheKey(second);
  assert(first_key != second_key);
  assert(first_key.size() == first.device_uuid.size() + 1 +
                               first.ipc_handle.size());
  assert(first_key[first.device_uuid.size()] == '\0');
  assert(first_key.substr(0, first.device_uuid.size()) == first.device_uuid);

  return 0;
}

