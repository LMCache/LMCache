// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nixl.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace lmcache {
namespace connector {

struct NixlTransferBuffer {
  std::string key;
  void* data;
  size_t length;
};

// Allocates a device id unique among all concurrently live storage
// descriptors of one worker context.
using NixlDescriptorIdAllocator = std::function<uint64_t()>;

// Preparation failure on a pending load: carries the per-key results that
// preserve the serial load semantics (false bits, successful tile completion).
class NixlLoadPrepareError : public std::runtime_error {
 public:
  NixlLoadPrepareError(const std::string& message,
                       std::vector<uint8_t> query_results)
      : std::runtime_error(message), results(std::move(query_results)) {}

  std::vector<uint8_t> results;
};

// Move-only (owned via unique_ptr) pending transfer that owns its
// registration, prepared dlists, request handle, and per-key results.
// Release order is always request -> dlists -> registration.
class NixlPendingOperation {
 public:
  NixlPendingOperation(const NixlPendingOperation&) = delete;
  NixlPendingOperation& operator=(const NixlPendingOperation&) = delete;
  virtual ~NixlPendingOperation() = default;

  // Submits and returns NIXL_IN_PROG, or NIXL_SUCCESS when already
  // complete (empty load or synchronous submit). Throws on failure.
  virtual nixl_status_t post() = 0;
  // Non-blocking: NIXL_IN_PROG while pending, otherwise terminal.
  virtual nixl_status_t poll() = 0;
  // discard(), then moves the latched per-key results (empty for stores).
  virtual std::vector<uint8_t> complete() = 0;
  // Allocation-free, throwing-free release in order request -> dlists ->
  // registration.
  virtual void discard() noexcept = 0;

 protected:
  NixlPendingOperation() = default;
};

struct NixlStorageCapabilities {
  bool query = true;
  bool deletion = false;
  bool direct_io = false;
  bool atomic_publication = false;
  // Required by the cross-call reactor, not sufficient: worker-loop
  // eligibility is decided once by nixl_reactor_eligible in reactor.h.
  bool pending_transfers = false;
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

  // Pending-path seams; the defaults throw for opted-out strategies.
  virtual std::unique_ptr<NixlPendingOperation> prepare_store(
      nixlAgent& agent, nixlBackendH* backend, const std::string& agent_name,
      const std::vector<NixlTransferBuffer>& buffers,
      const NixlDescriptorIdAllocator& allocate_id) {
    (void)agent;
    (void)backend;
    (void)agent_name;
    (void)buffers;
    (void)allocate_id;
    throw std::runtime_error("NIXL storage does not support pending transfers");
  }
  virtual std::unique_ptr<NixlPendingOperation> prepare_load(
      nixlAgent& agent, nixlBackendH* backend, const std::string& agent_name,
      const std::vector<NixlTransferBuffer>& buffers,
      const NixlDescriptorIdAllocator& allocate_id) {
    (void)agent;
    (void)backend;
    (void)agent_name;
    (void)buffers;
    (void)allocate_id;
    throw std::runtime_error("NIXL storage does not support pending transfers");
  }
  // One-shot fallback after a failed load: returns a retry over present
  // keys (empty when none). NixlLoadPrepareError completes the tile with
  // false bits; any other throw fails it.
  virtual std::unique_ptr<NixlPendingOperation> prepare_load_fallback(
      nixlAgent& agent, nixlBackendH* backend, const std::string& agent_name,
      const std::vector<NixlTransferBuffer>& buffers,
      const NixlDescriptorIdAllocator& allocate_id) {
    (void)agent;
    (void)backend;
    (void)agent_name;
    (void)buffers;
    (void)allocate_id;
    throw std::runtime_error("NIXL storage does not support pending transfers");
  }
};

std::unique_ptr<NixlStorageStrategy> make_nixl_storage_strategy(
    NixlStorageKind storage_kind,
    const std::unordered_map<std::string, std::string>& backend_params);

const char* nixl_storage_kind_name(NixlStorageKind storage_kind);

}  // namespace connector
}  // namespace lmcache
