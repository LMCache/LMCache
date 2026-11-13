// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nixl.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../connector_base.h"
#include "storage.h"

namespace lmcache {
namespace connector {

struct WorkerNixlContext {
  WorkerNixlContext(
      std::string agent_name, const std::string& backend_name,
      const std::unordered_map<std::string, std::string>& backend_params,
      uintptr_t l1_base, size_t l1_size, size_t l1_alignment);
  ~WorkerNixlContext();

  WorkerNixlContext(const WorkerNixlContext&) = delete;
  WorkerNixlContext& operator=(const WorkerNixlContext&) = delete;

  std::string agent_name;
  std::unique_ptr<nixlAgent> agent;
  nixlBackendH* backend = nullptr;
  NixlStorageKind storage_kind = NixlStorageKind::File;
  nixl_reg_dlist_t l1_registration;
  bool l1_registered = false;
  std::unique_ptr<NixlStorageStrategy> storage;
};

class NixlConnector : public ConnectorBase<std::unique_ptr<WorkerNixlContext>> {
 public:
  NixlConnector(std::string backend,
                std::unordered_map<std::string, std::string> backend_params,
                int num_workers, uintptr_t l1_base, size_t l1_size,
                size_t l1_alignment);
  ~NixlConnector() override;

  const std::string& storage_type() const noexcept;
  bool supports_query() const noexcept;
  bool supports_delete() const noexcept;
  bool supports_direct_io() const noexcept;
  bool atomic_publication() const noexcept;

 protected:
  std::unique_ptr<WorkerNixlContext> create_connection() override;
  void do_single_get(std::unique_ptr<WorkerNixlContext>& context,
                     const std::string& key, void* buf, size_t len,
                     size_t chunk_size) override;
  void do_single_set(std::unique_ptr<WorkerNixlContext>& context,
                     const std::string& key, const void* buf, size_t len,
                     size_t chunk_size) override;
  bool do_single_exists(std::unique_ptr<WorkerNixlContext>& context,
                        const std::string& key) override;
  bool do_single_delete(std::unique_ptr<WorkerNixlContext>& context,
                        const std::string& key) override;
  void do_batch_get(std::unique_ptr<WorkerNixlContext>& context,
                    const Request& request) override;
  void do_batch_set(std::unique_ptr<WorkerNixlContext>& context,
                    const Request& request) override;
  void do_batch_exists(std::unique_ptr<WorkerNixlContext>& context,
                       const Request& request) override;
  void do_batch_delete(std::unique_ptr<WorkerNixlContext>& context,
                       const Request& request) override;

 private:
  std::vector<NixlTransferBuffer> validate_buffers(
      const Request& request) const;
  void validate_buffer(const void* buffer, size_t length) const;

  uintptr_t l1_base_;
  size_t l1_size_;
  size_t l1_alignment_;
  std::string storage_type_;
  NixlStorageCapabilities storage_capabilities_;
  std::mutex contexts_mutex_;
  std::vector<std::unique_ptr<WorkerNixlContext>> contexts_;
};

}  // namespace connector
}  // namespace lmcache
