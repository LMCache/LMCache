// SPDX-License-Identifier: Apache-2.0

#include "connector.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace lmcache {
namespace connector {

MooncakeConnector::MooncakeConnector(ConfigDict config, int num_workers)
    : ConnectorBase(num_workers), config_(std::move(config)) {
  // Create a RealClient via the static factory.
  client_ = mooncake::RealClient::create();
  if (!client_) {
    throw std::runtime_error("Failed to create mooncake RealClient");
  }

  // Forward the config dict to setup_internal().
  mooncake::ConfigDict mc_config(config_.begin(), config_.end());
  auto result = client_->setup_internal(mc_config);
  if (!result.has_value()) {
    throw std::runtime_error("Mooncake setup_internal failed");
  }

  start_workers();  // IMPORTANT: call at END of ctor
}

MooncakeConnector::~MooncakeConnector() {
  close();
  if (client_) {
    client_->tearDownAll();
    client_.reset();
  }
}

WorkerMooncakeConn MooncakeConnector::create_connection() {
  WorkerMooncakeConn conn;
  conn.client = client_.get();
  return conn;
}

void MooncakeConnector::do_single_get(WorkerMooncakeConn& conn,
                                      const std::string& key, void* buf,
                                      size_t len, size_t chunk_size) {
  int64_t bytes_read = conn.client->get_into(key, buf, len);
  if (bytes_read < 0) {
    throw std::runtime_error("Mooncake get_into failed for key: " + key);
  }
}

void MooncakeConnector::do_single_set(WorkerMooncakeConn& conn,
                                      const std::string& key, const void* buf,
                                      size_t len, size_t chunk_size) {
  int rc = conn.client->put_from(key, const_cast<void*>(buf), len);
  if (rc != 0) {
    throw std::runtime_error("Mooncake put_from failed for key: " + key);
  }
}

bool MooncakeConnector::do_single_exists(WorkerMooncakeConn& conn,
                                         const std::string& key) {
  // isExist returns: 1=exists, 0=not, -1=error
  int result = conn.client->isExist(key);
  if (result < 0) {
    throw std::runtime_error("Mooncake isExist failed for key: " + key);
  }
  return result == 1;
}

}  // namespace connector
}  // namespace lmcache
