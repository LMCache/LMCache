// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../connector_base.h"
#include "real_client.h"

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
  mooncake::RealClient* client{nullptr};
};

class MooncakeConnector : public ConnectorBase<WorkerMooncakeConn> {
 public:
  MooncakeConnector(ConfigDict config, int num_workers);
  ~MooncakeConnector() override;

 protected:
  WorkerMooncakeConn create_connection() override;

  void do_single_get(WorkerMooncakeConn& conn, const std::string& key,
                     void* buf, size_t len, size_t chunk_size) override;

  void do_single_set(WorkerMooncakeConn& conn, const std::string& key,
                     const void* buf, size_t len, size_t chunk_size) override;

  bool do_single_exists(WorkerMooncakeConn& conn,
                        const std::string& key) override;

 private:
  // Shared Mooncake RealClient instance.
  std::shared_ptr<mooncake::RealClient> client_;

  // The original config dict (kept for diagnostics).
  ConfigDict config_;
};

}  // namespace connector
}  // namespace lmcache