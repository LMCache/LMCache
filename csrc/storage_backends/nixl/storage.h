// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nixl.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace lmcache {
namespace connector {

struct NixlTransferBuffer {
  std::string key;
  void* data;
  size_t length;
};

struct NixlStorageCapabilities {
  bool query = true;
  bool deletion = false;
  bool direct_io = false;
  bool atomic_publication = false;
};

enum class NixlStorageKind { File, Object };

class NixlStorageStrategy {
 public:
  virtual ~NixlStorageStrategy() = default;

  virtual void store(nixlAgent& agent, nixlBackendH* backend,
                     const std::string& agent_name,
                     const std::vector<NixlTransferBuffer>& buffers,
                     const std::atomic<bool>& stop) = 0;
  virtual std::vector<uint8_t> load(
      nixlAgent& agent, nixlBackendH* backend, const std::string& agent_name,
      const std::vector<NixlTransferBuffer>& buffers,
      const std::atomic<bool>& stop) = 0;
  virtual std::vector<uint8_t> exists(nixlAgent& agent, nixlBackendH* backend,
                                      const std::vector<std::string>& keys) = 0;
  virtual std::vector<uint8_t> remove(const std::vector<std::string>& keys) = 0;
  virtual NixlStorageCapabilities capabilities() const = 0;
};

std::unique_ptr<NixlStorageStrategy> make_nixl_storage_strategy(
    NixlStorageKind storage_kind,
    const std::unordered_map<std::string, std::string>& backend_params,
    size_t l1_alignment);

const char* nixl_storage_kind_name(NixlStorageKind storage_kind);

std::string nixl_persistent_identity(const std::string& serialized_key,
                                     bool shard_directories);

}  // namespace connector
}  // namespace lmcache
