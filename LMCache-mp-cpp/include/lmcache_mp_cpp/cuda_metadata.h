// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "lmcache_mp_cpp/cuda_transfer.h"

#include <cstdint>
#include <optional>
#include <string>

namespace lmcache::mp {

std::optional<std::uint64_t> DtypeElementSize(const std::string& dtype);

std::string IpcHandleCacheKey(const KvTensorMetadata& metadata);

}  // namespace lmcache::mp

