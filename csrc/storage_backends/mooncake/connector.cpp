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

namespace {

bool range_contains(const RegisteredMemoryRegion& region, const void* buf,
                    size_t len) {
  const auto region_begin = reinterpret_cast<std::uintptr_t>(region.base);
  const auto region_end = region_begin + region.size;
  const auto buf_begin = reinterpret_cast<std::uintptr_t>(buf);
  const auto buf_end = buf_begin + len;
  return region_begin <= buf_begin && buf_end <= region_end;
}

}  // namespace

MooncakeConnector::MooncakeConnector(ConfigDict config, int num_workers,
                                     std::uintptr_t preregister_l1_base,
                                     size_t preregister_l1_size)
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

  preregister_l1_memory(preregister_l1_base, preregister_l1_size);

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

void MooncakeConnector::close() {
  ConnectorBase<WorkerMooncakeConn>::close();
  unregister_all_buffers();
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

void MooncakeConnector::ensure_registered(const void* buf, size_t len) {
  if (buf == nullptr) {
    throw std::runtime_error(
        "Mooncake buffer registration failed: null buffer");
  }

  if (is_within_registered_region(buf, len)) {
    return;
  }

  void* mutable_buf = const_cast<void*>(buf);
  while (true) {
    std::unique_lock<std::mutex> lock(registered_buffers_mu_);
    auto it = registered_buffers_.find(buf);
    while (it != registered_buffers_.end() && it->second.registering) {
      registered_buffers_cv_.wait(lock);
      it = registered_buffers_.find(buf);
    }

    if (it != registered_buffers_.end() && it->second.size >= len) {
      return;
    }

    if (it == registered_buffers_.end()) {
      RegisteredBufferState state;
      state.registering = true;
      registered_buffers_.emplace(buf, state);
      break;
    }

    throw std::runtime_error(
        "Mooncake lazy registration does not support resizing an existing "
        "buffer registration (existing_size=" +
        std::to_string(it->second.size) + ", requested_size=" +
        std::to_string(len) +
        "); use a stable buffer size or preregister L1 memory.");
  }

  int register_rc = client_->register_buffer(mutable_buf, len);
  {
    std::lock_guard<std::mutex> guard(registered_buffers_mu_);
    auto it = registered_buffers_.find(buf);
    if (register_rc != 0) {
      if (it != registered_buffers_.end()) {
        registered_buffers_.erase(it);
      }
    } else if (it != registered_buffers_.end()) {
      it->second.size = len;
      it->second.registering = false;
    }
  }
  registered_buffers_cv_.notify_all();

  if (register_rc != 0) {
    throw std::runtime_error("Mooncake register_buffer failed");
  }
}

void MooncakeConnector::preregister_l1_memory(std::uintptr_t base,
                                              size_t size) {
  if (base == 0 || size == 0) {
    return;
  }

  const size_t max_registration_size =
      mooncake::globalConfig().max_mr_size == 0
          ? size
          : mooncake::globalConfig().max_mr_size;
  size_t remaining = size;
  auto current = base;
  std::vector<RegisteredMemoryRegion> newly_registered_regions;

  while (remaining > 0) {
    const size_t segment_size = std::min(remaining, max_registration_size);
    void* segment_ptr = reinterpret_cast<void*>(current);
    const int register_rc = client_->register_buffer(segment_ptr, segment_size);
    if (register_rc != 0) {
      for (auto it = newly_registered_regions.rbegin();
           it != newly_registered_regions.rend(); ++it) {
        client_->unregister_buffer(const_cast<void*>(it->base));
      }
      throw std::runtime_error("Mooncake preregister_l1_memory failed");
    }

    newly_registered_regions.push_back({segment_ptr, segment_size});
    current += segment_size;
    remaining -= segment_size;
  }

  preregistered_regions_.insert(preregistered_regions_.end(),
                                newly_registered_regions.begin(),
                                newly_registered_regions.end());
}

bool MooncakeConnector::is_within_registered_region(const void* buf,
                                                    size_t len) const {
  for (const auto& region : preregistered_regions_) {
    if (range_contains(region, buf, len)) {
      return true;
    }
  }
  return false;
}

void MooncakeConnector::unregister_all_buffers() noexcept {
  std::unordered_map<const void*, RegisteredBufferState> buffers_to_unregister;
  std::vector<RegisteredMemoryRegion> regions_to_unregister;
  {
    std::lock_guard<std::mutex> guard(registered_buffers_mu_);
    if (registered_buffers_.empty() && preregistered_regions_.empty()) {
      return;
    }
    buffers_to_unregister.swap(registered_buffers_);
    regions_to_unregister.swap(preregistered_regions_);
  }

  for (const auto& [buf, state] : buffers_to_unregister) {
    if (state.registering || state.size == 0) {
      continue;
    }
    if (client_ == nullptr) {
      break;
    }
    client_->unregister_buffer(const_cast<void*>(buf));
  }

  for (auto it = regions_to_unregister.rbegin();
       it != regions_to_unregister.rend(); ++it) {
    if (client_ == nullptr) {
      break;
    }
    client_->unregister_buffer(const_cast<void*>(it->base));
  }
}

}  // namespace connector
}  // namespace lmcache
