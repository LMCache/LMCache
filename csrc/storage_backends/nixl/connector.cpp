// SPDX-License-Identifier: Apache-2.0

#include "connector.h"

#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace lmcache {
namespace connector {
namespace {

std::atomic<uint64_t> next_agent_id{1};

void check_nixl(nixl_status_t status, const std::string& operation) {
  if (status != NIXL_SUCCESS) {
    throw std::runtime_error(operation +
                             " failed: " + nixlEnumStrings::statusStr(status));
  }
}

bool contains_mem_type(const nixl_mem_list_t& memory_types,
                       nixl_mem_t expected) {
  return std::find(memory_types.begin(), memory_types.end(), expected) !=
         memory_types.end();
}

std::string make_agent_name() {
  return "lmcache-nixl-" + std::to_string(static_cast<long long>(getpid())) +
         "-" + std::to_string(next_agent_id.fetch_add(1));
}

}  // namespace

WorkerNixlContext::WorkerNixlContext(
    std::string context_agent_name, const std::string& backend_name,
    const std::unordered_map<std::string, std::string>& backend_params,
    uintptr_t l1_base, size_t l1_size, size_t l1_alignment)
    : agent_name(std::move(context_agent_name)), l1_registration(DRAM_SEG) {
  nixlAgentConfig config;
  config.syncMode = nixl_thread_sync_t::NIXL_THREAD_SYNC_NONE;
  agent = std::make_unique<nixlAgent>(agent_name, config);

  nixl_mem_list_t supported_memory;
  nixl_b_params_t plugin_defaults;
  check_nixl(
      agent->getPluginParams(backend_name, supported_memory, plugin_defaults),
      "NIXL plugin discovery for " + backend_name);
  if (!contains_mem_type(supported_memory, DRAM_SEG)) {
    throw std::runtime_error("NIXL backend " + backend_name +
                             " does not support required DRAM_SEG memory");
  }
  bool supports_file = contains_mem_type(supported_memory, FILE_SEG);
  bool supports_object = contains_mem_type(supported_memory, OBJ_SEG);
  if (supports_file == supports_object) {
    std::string supported_storage = supports_file
                                        ? "both FILE_SEG and OBJ_SEG"
                                        : "neither FILE_SEG nor OBJ_SEG";
    throw std::runtime_error("NIXL backend " + backend_name + " supports " +
                             supported_storage +
                             "; exactly one storage segment is required");
  }
  storage_kind =
      supports_file ? NixlStorageKind::File : NixlStorageKind::Object;

  nixl_b_params_t params(backend_params.begin(), backend_params.end());
  check_nixl(agent->createBackend(backend_name, params, backend),
             "NIXL backend creation for " + backend_name);
  if (backend == nullptr) {
    throw std::runtime_error("NIXL returned a null backend handle for " +
                             backend_name);
  }

  l1_registration.addDesc(nixlBlobDesc(l1_base, l1_size, 0, std::string()));
  nixl_opt_args_t options;
  options.backends = {backend};
  check_nixl(agent->registerMem(l1_registration, &options),
             "NIXL L1 DRAM registration");
  l1_registered = true;

  try {
    storage =
        make_nixl_storage_strategy(storage_kind, backend_params, l1_alignment);
  } catch (...) {
    agent->deregisterMem(l1_registration, &options);
    l1_registered = false;
    throw;
  }
}

WorkerNixlContext::~WorkerNixlContext() {
  storage.reset();
  if (agent && l1_registered) {
    nixl_opt_args_t options;
    options.backends = {backend};
    nixl_status_t status = agent->deregisterMem(l1_registration, &options);
    if (status != NIXL_SUCCESS) {
      std::fprintf(stderr, "[LMCache NIXL] L1 deregistration failed: %s\n",
                   nixlEnumStrings::statusStr(status).c_str());
    }
    l1_registered = false;
  }
}

NixlConnector::NixlConnector(
    std::string backend,
    std::unordered_map<std::string, std::string> backend_params,
    int num_workers, uintptr_t l1_base, size_t l1_size, size_t l1_alignment)
    : ConnectorBase(num_workers),
      l1_base_(l1_base),
      l1_size_(l1_size),
      l1_alignment_(l1_alignment) {
  if (backend.empty()) {
    throw std::runtime_error("backend must not be empty");
  }
  if (l1_base_ == 0 || l1_size_ == 0 || l1_alignment_ == 0) {
    throw std::runtime_error(
        "L1 base, size, and alignment must all be positive");
  }
  if (l1_base_ % l1_alignment_ != 0) {
    throw std::runtime_error("L1 base is not aligned to l1_alignment");
  }
  if (l1_size_ % l1_alignment_ != 0) {
    throw std::runtime_error("L1 size is not aligned to l1_alignment");
  }
  if (l1_size_ > std::numeric_limits<uintptr_t>::max() - l1_base_) {
    throw std::runtime_error("L1 arena address range overflows uintptr_t");
  }

  contexts_.reserve(static_cast<size_t>(num_workers));
  for (int worker = 0; worker < num_workers; ++worker) {
    std::unique_ptr<WorkerNixlContext> context =
        std::make_unique<WorkerNixlContext>(make_agent_name(), backend,
                                            backend_params, l1_base_, l1_size_,
                                            l1_alignment_);
    if (contexts_.empty()) {
      storage_type_ = nixl_storage_kind_name(context->storage_kind);
      storage_capabilities_ = context->storage->capabilities();
    } else if (context->storage_kind != contexts_.front()->storage_kind) {
      throw std::runtime_error(
          "NIXL backend reported inconsistent storage capabilities");
    }
    contexts_.push_back(std::move(context));
  }
  start_workers();
}

NixlConnector::~NixlConnector() { close(); }

const std::string& NixlConnector::storage_type() const noexcept {
  return storage_type_;
}

bool NixlConnector::supports_query() const noexcept {
  return storage_capabilities_.query;
}

bool NixlConnector::supports_delete() const noexcept {
  return storage_capabilities_.deletion;
}

bool NixlConnector::supports_direct_io() const noexcept {
  return storage_capabilities_.direct_io;
}

bool NixlConnector::atomic_publication() const noexcept {
  return storage_capabilities_.atomic_publication;
}

std::unique_ptr<WorkerNixlContext> NixlConnector::create_connection() {
  std::lock_guard<std::mutex> lock(contexts_mutex_);
  if (contexts_.empty()) {
    throw std::runtime_error("no prepared NIXL worker context is available");
  }
  std::unique_ptr<WorkerNixlContext> context = std::move(contexts_.back());
  contexts_.pop_back();
  return context;
}

void NixlConnector::do_single_get(std::unique_ptr<WorkerNixlContext>& context,
                                  const std::string& key, void* buffer,
                                  size_t length, size_t chunk_size) {
  (void)chunk_size;
  validate_buffer(buffer, length);
  std::vector<NixlTransferBuffer> buffers{{key, buffer, length}};
  if (!context->storage
           ->load(*context->agent, context->backend, context->agent_name,
                  buffers, stop_)
           .front()) {
    throw std::runtime_error("NIXL key was not loaded: " + key);
  }
}

void NixlConnector::do_single_set(std::unique_ptr<WorkerNixlContext>& context,
                                  const std::string& key, const void* buffer,
                                  size_t length, size_t chunk_size) {
  (void)chunk_size;
  validate_buffer(buffer, length);
  std::vector<NixlTransferBuffer> buffers{
      {key, const_cast<void*>(buffer), length}};
  context->storage->store(*context->agent, context->backend,
                          context->agent_name, buffers, stop_);
}

bool NixlConnector::do_single_exists(
    std::unique_ptr<WorkerNixlContext>& context, const std::string& key) {
  return context->storage->exists(*context->agent, context->backend, {key})
      .front();
}

bool NixlConnector::do_single_delete(
    std::unique_ptr<WorkerNixlContext>& context, const std::string& key) {
  return context->storage->remove({key}).front();
}

void NixlConnector::do_batch_get(std::unique_ptr<WorkerNixlContext>& context,
                                 const Request& request) {
  std::vector<NixlTransferBuffer> buffers = validate_buffers(request);
  std::vector<uint8_t> results = context->storage->load(
      *context->agent, context->backend, context->agent_name, buffers, stop_);
  if (results.size() != request.keys.size()) {
    throw std::runtime_error("NIXL load returned an invalid result count");
  }
  for (size_t index = 0; index < results.size(); ++index) {
    request.batch->per_key_results[request.start_idx + index] = results[index];
  }
}

void NixlConnector::do_batch_set(std::unique_ptr<WorkerNixlContext>& context,
                                 const Request& request) {
  std::vector<NixlTransferBuffer> buffers = validate_buffers(request);
  context->storage->store(*context->agent, context->backend,
                          context->agent_name, buffers, stop_);
}

void NixlConnector::do_batch_exists(std::unique_ptr<WorkerNixlContext>& context,
                                    const Request& request) {
  std::vector<uint8_t> results =
      context->storage->exists(*context->agent, context->backend, request.keys);
  if (results.size() != request.keys.size()) {
    throw std::runtime_error("NIXL query returned an invalid result count");
  }
  for (size_t index = 0; index < results.size(); ++index) {
    request.batch->per_key_results[request.start_idx + index] = results[index];
  }
}

void NixlConnector::do_batch_delete(std::unique_ptr<WorkerNixlContext>& context,
                                    const Request& request) {
  std::vector<uint8_t> results = context->storage->remove(request.keys);
  if (results.size() != request.keys.size()) {
    throw std::runtime_error("NIXL delete returned an invalid result count");
  }
  for (size_t index = 0; index < results.size(); ++index) {
    request.batch->per_key_results[request.start_idx + index] = results[index];
  }
}

std::vector<NixlTransferBuffer> NixlConnector::validate_buffers(
    const Request& request) const {
  if (request.keys.size() != request.buf_ptrs.size() ||
      request.keys.size() != request.buf_lens.size()) {
    throw std::runtime_error("invalid native NIXL request vectors");
  }
  std::vector<NixlTransferBuffer> buffers;
  buffers.reserve(request.keys.size());
  for (size_t index = 0; index < request.keys.size(); ++index) {
    validate_buffer(request.buf_ptrs[index], request.buf_lens[index]);
    buffers.push_back({request.keys[index], request.buf_ptrs[index],
                       request.buf_lens[index]});
  }
  return buffers;
}

void NixlConnector::validate_buffer(const void* buffer, size_t length) const {
  uintptr_t address = reinterpret_cast<uintptr_t>(buffer);
  if (address < l1_base_) {
    throw std::runtime_error("buffer begins below the registered L1 arena");
  }
  uintptr_t offset = address - l1_base_;
  if (offset > l1_size_ || length > l1_size_ - offset) {
    throw std::runtime_error("buffer extends beyond the registered L1 arena");
  }
  if (length == 0) {
    throw std::runtime_error("buffer length must be positive");
  }
  if (address % l1_alignment_ != 0 || length % l1_alignment_ != 0) {
    throw std::runtime_error("buffer address and length must be L1-aligned");
  }
}

}  // namespace connector
}  // namespace lmcache
