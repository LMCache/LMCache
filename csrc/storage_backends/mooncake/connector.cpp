// SPDX-License-Identifier: Apache-2.0

// Standard
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

// Third Party
#include "config.h"

// Local
#include "connector.h"

namespace lmcache {
namespace connector {

MooncakeConnector::MooncakeConnector(ConfigDict config, int num_workers,
                                     L1RegistrationConfig l1_registration)
    : ConnectorBase(num_workers),
      config_(std::move(config)),
      l1_registration_(l1_registration) {
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

  if (l1_registration.is_valid()) {
    preregister_l1_memory(l1_registration.base, l1_registration.size);
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
  (void)chunk_size;
  ensure_registered(buf, len);
  int64_t bytes_read = conn.client->get_into(key, buf, len);
  if (bytes_read < 0) {
    throw std::runtime_error("Mooncake get_into failed for key: " + key);
  }
}

void MooncakeConnector::do_single_set(WorkerMooncakeConn& conn,
                                      const std::string& key, const void* buf,
                                      size_t len, size_t chunk_size) {
  (void)chunk_size;
  ensure_registered(buf, len);
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

void MooncakeConnector::on_workers_stopped() { unregister_all_buffers(); }

void MooncakeConnector::ensure_registered(const void* buf, size_t len) {
  if (!l1_registration_.is_valid()) {
    return;
  }
  if (buf == nullptr) {
    throw std::runtime_error(
        "Mooncake buffer registration failed: null buffer");
  }
  if (len == 0) {
    throw std::runtime_error(
        "Mooncake buffer registration failed: zero length");
  }

  if (preregistered_block_size_ == 0) {
    throw std::runtime_error("Mooncake preregistered block size is invalid");
  }

  const auto registered_begin = l1_registration_.base;
  const auto registered_end = registered_begin + l1_registration_.size;
  const auto buf_begin = reinterpret_cast<std::uintptr_t>(buf);
  const auto buf_end = buf_begin + len;

  if (buf_begin < registered_begin || buf_end > registered_end) {
    throw std::runtime_error(
        "Buffer is outside of preregistered L1 region; Mooncake lazy "
        "registration is disabled");
  }

  const auto relative_begin = buf_begin - registered_begin;
  const auto relative_end = (buf_end - 1) - registered_begin;
  const size_t left_region = relative_begin / preregistered_block_size_;
  const size_t right_region = relative_end / preregistered_block_size_;

  if (left_region != right_region) {
    throw std::runtime_error("Buffer crosses preregistered Mooncake regions");
  }
}

void MooncakeConnector::preregister_l1_memory(std::uintptr_t base,
                                              size_t size) {
  if (base == 0 || size == 0) {
    return;
  }

  preregistered_block_size_ = mooncake::globalConfig().max_mr_size == 0
                                  ? size
                                  : mooncake::globalConfig().max_mr_size;
  const size_t max_registration_size = preregistered_block_size_;
  size_t remaining = size;
  auto current = base;

  while (remaining > 0) {
    const size_t segment_size = std::min(remaining, max_registration_size);
    void* segment_ptr = reinterpret_cast<void*>(current);
    const int register_rc = client_->register_buffer(segment_ptr, segment_size);
    if (register_rc != 0) {
      auto rollback_remaining = current - base;
      while (rollback_remaining > 0) {
        const auto rollback_size =
            std::min(rollback_remaining, max_registration_size);
        rollback_remaining -= rollback_size;
        void* rollback_ptr = reinterpret_cast<void*>(base + rollback_remaining);
        try {
          client_->unregister_buffer(rollback_ptr);
        } catch (...) {
          // Keep rolling back the remaining segments so a cleanup failure does
          // not mask the original registration error or strand later regions.
        }
      }
      preregistered_block_size_ = 0;
      throw std::runtime_error("Mooncake preregister_l1_memory failed");
    }

    current += segment_size;
    remaining -= segment_size;
  }
}

void MooncakeConnector::unregister_all_buffers() noexcept {
  if (client_ == nullptr || !l1_registration_.is_valid() ||
      preregistered_block_size_ == 0) {
    return;
  }

  auto current = l1_registration_.base;
  size_t remaining = l1_registration_.size;
  while (remaining > 0) {
    const auto segment_size = std::min(remaining, preregistered_block_size_);
    void* segment_ptr = reinterpret_cast<void*>(current);
    try {
      client_->unregister_buffer(segment_ptr);
    } catch (...) {
      // Preserve noexcept during teardown and keep attempting the remaining
      // segments so earlier failures do not strand later registrations.
    }
    current += segment_size;
    remaining -= segment_size;
  }

  preregistered_block_size_ = 0;
}

}  // namespace connector
}  // namespace lmcache
