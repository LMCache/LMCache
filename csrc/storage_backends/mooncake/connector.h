// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"
#include "real_client.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace lmcache {
namespace connector {

// ConfigDict mirrors mooncake::ConfigDict
// (std::unordered_map<std::string, std::string>).
using ConfigDict = std::unordered_map<std::string, std::string>;

// Per-worker connection state for the Mooncake connector.
// Each worker holds a raw pointer to the shared
// RealClient (owned by MooncakeConnector).
struct WorkerMooncakeConn {
  mooncake::PyClient* client{nullptr};
};

struct RegisteredMemoryRegion {
  const void* base{nullptr};
  size_t size{0};
};

class MooncakeConnector : public ConnectorBase<WorkerMooncakeConn> {
 public:
  MooncakeConnector(ConfigDict config, int num_workers,
                    std::uintptr_t preregister_l1_base = 0,
                    size_t preregister_l1_size = 0);
  MooncakeConnector(std::shared_ptr<mooncake::PyClient> client,
                    int num_workers,
                    std::uintptr_t preregister_l1_base = 0,
                    size_t preregister_l1_size = 0);
  ~MooncakeConnector() override;

  void close() override;

 protected:
  WorkerMooncakeConn create_connection() override;

  void do_single_get(WorkerMooncakeConn& conn, const std::string& key,
                     void* buf, size_t len, size_t chunk_size) override;

  void do_single_set(WorkerMooncakeConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;

  bool do_single_exists(WorkerMooncakeConn& conn,
                        const std::string& key) override;

 private:
  void ensure_registered(const void* buf, size_t len);
  void preregister_l1_memory(std::uintptr_t base, size_t size);
  bool is_within_registered_region(const void* buf, size_t len) const;
  void unregister_all_buffers() noexcept;
  // Future extension point: use a pre-registered staging pool when
  // buffers are too ephemeral to benefit from direct lazy registration.
  void copy_to_registered_staging_and_put() = delete;

  // Shared Mooncake RealClient instance.
  std::shared_ptr<mooncake::PyClient> client_;
  std::shared_ptr<mooncake::RealClient> owned_real_client_;

  // The original config dict (kept for diagnostics).
  ConfigDict config_;

  std::mutex registered_buffers_mu_;
  std::unordered_map<const void*, size_t> registered_buffers_;
  std::vector<RegisteredMemoryRegion> preregistered_regions_;
};

}  // namespace connector
}  // namespace lmcache
