// SPDX-License-Identifier: Apache-2.0

#include "storage.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>

namespace lmcache {
namespace connector {
namespace {

constexpr char kKeySeparator = '@';
constexpr const char* kFileExtension = ".bin";
std::atomic<uint64_t> next_temporary_id{1};

void check_nixl(nixl_status_t status, const std::string& operation) {
  if (status != NIXL_SUCCESS) {
    throw std::runtime_error(operation +
                             " failed: " + nixlEnumStrings::statusStr(status));
  }
}

class UniqueFd {
 public:
  explicit UniqueFd(int fd = -1) : fd_(fd) {}
  ~UniqueFd() { reset(); }

  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.release()) {}
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) reset(other.release());
    return *this;
  }

  int get() const { return fd_; }
  int release() {
    int result = fd_;
    fd_ = -1;
    return result;
  }
  void reset(int fd = -1) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = fd;
  }

 private:
  int fd_;
};

class TemporaryFile {
 public:
  TemporaryFile(std::filesystem::path path, int fd)
      : path_(std::move(path)), fd_(fd) {}
  ~TemporaryFile() {
    fd_.reset();
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  TemporaryFile(const TemporaryFile&) = delete;
  TemporaryFile& operator=(const TemporaryFile&) = delete;
  TemporaryFile(TemporaryFile&&) noexcept = default;
  TemporaryFile& operator=(TemporaryFile&&) noexcept = default;

  int fd() const { return fd_.get(); }
  const std::filesystem::path& path() const { return path_; }
  void close() { fd_.reset(); }

 private:
  std::filesystem::path path_;
  UniqueFd fd_;
};

class RegisteredMemory {
 public:
  RegisteredMemory(nixlAgent& agent, nixlBackendH* backend,
                   nixl_reg_dlist_t descriptors)
      : agent_(agent), descriptors_(std::move(descriptors)) {
    options_.backends = {backend};
    check_nixl(agent_.registerMem(descriptors_, &options_),
               "NIXL storage registration");
    registered_ = true;
  }

  ~RegisteredMemory() {
    if (registered_) {
      nixl_status_t status = agent_.deregisterMem(descriptors_, &options_);
      if (status != NIXL_SUCCESS) {
        std::fprintf(stderr,
                     "[LMCache NIXL] storage deregistration failed: %s\n",
                     nixlEnumStrings::statusStr(status).c_str());
      }
    }
  }

  RegisteredMemory(const RegisteredMemory&) = delete;
  RegisteredMemory& operator=(const RegisteredMemory&) = delete;

 private:
  nixlAgent& agent_;
  nixl_reg_dlist_t descriptors_;
  nixl_opt_args_t options_;
  bool registered_ = false;
};

class PreparedDlist {
 public:
  PreparedDlist(nixlAgent& agent, nixlBackendH* backend,
                const nixl_xfer_dlist_t& descriptors)
      : agent_(agent) {
    nixl_opt_args_t options;
    options.backends = {backend};
    check_nixl(agent_.prepXferDlist(descriptors, handle_, &options),
               "NIXL local descriptor preparation");
  }

  PreparedDlist(nixlAgent& agent, nixlBackendH* backend,
                const std::string& agent_name,
                const nixl_xfer_dlist_t& descriptors)
      : agent_(agent) {
    nixl_opt_args_t options;
    options.backends = {backend};
    check_nixl(agent_.prepXferDlist(agent_name, descriptors, handle_, &options),
               "NIXL storage descriptor preparation");
  }

  ~PreparedDlist() {
    if (handle_ != nullptr) {
      nixl_status_t status = agent_.releasedDlistH(handle_);
      if (status != NIXL_SUCCESS) {
        std::fprintf(stderr, "[LMCache NIXL] descriptor release failed: %s\n",
                     nixlEnumStrings::statusStr(status).c_str());
      }
    }
  }

  PreparedDlist(const PreparedDlist&) = delete;
  PreparedDlist& operator=(const PreparedDlist&) = delete;
  nixlDlistH* get() const { return handle_; }

 private:
  nixlAgent& agent_;
  nixlDlistH* handle_ = nullptr;
};

class TransferRequest {
 public:
  TransferRequest(nixlAgent& agent, nixlBackendH* backend,
                  nixl_xfer_op_t operation, const PreparedDlist& local,
                  const PreparedDlist& storage, size_t descriptor_count)
      : agent_(agent) {
    std::vector<int> indices(descriptor_count);
    std::iota(indices.begin(), indices.end(), 0);
    nixl_opt_args_t options;
    options.backends = {backend};
    check_nixl(agent_.makeXferReq(operation, local.get(), indices,
                                  storage.get(), indices, handle_, &options),
               "NIXL transfer creation");
  }

  ~TransferRequest() {
    if (handle_ != nullptr) {
      nixl_status_t status = agent_.releaseXferReq(handle_);
      if (status != NIXL_SUCCESS) {
        std::fprintf(stderr, "[LMCache NIXL] transfer release failed: %s\n",
                     nixlEnumStrings::statusStr(status).c_str());
      }
    }
  }

  TransferRequest(const TransferRequest&) = delete;
  TransferRequest& operator=(const TransferRequest&) = delete;

  void execute(const std::atomic<bool>& stop) {
    nixl_status_t status = agent_.postXferReq(handle_);
    if (status < NIXL_SUCCESS) {
      check_nixl(status, "NIXL transfer submission");
    }
    while (status == NIXL_IN_PROG) {
      if (stop.load(std::memory_order_acquire)) {
        throw std::runtime_error("NIXL transfer cancelled during shutdown");
      }
      status = agent_.getXferStatus(handle_);
      if (status == NIXL_IN_PROG) std::this_thread::yield();
    }
    check_nixl(status, "NIXL transfer progress");
  }

 private:
  nixlAgent& agent_;
  nixlXferReqH* handle_ = nullptr;
};

void execute_transfer(nixlAgent& agent, nixlBackendH* backend,
                      const std::string& agent_name, nixl_xfer_op_t operation,
                      const nixl_xfer_dlist_t& local_descriptors,
                      const nixl_reg_dlist_t& storage_registration,
                      const nixl_xfer_dlist_t& storage_descriptors,
                      const std::atomic<bool>& stop) {
  if (local_descriptors.isEmpty()) return;
  if (local_descriptors.descCount() != storage_descriptors.descCount()) {
    throw std::runtime_error("NIXL descriptor lists have different sizes");
  }
  RegisteredMemory registration(agent, backend, storage_registration);
  PreparedDlist local(agent, backend, local_descriptors);
  PreparedDlist storage(agent, backend, agent_name, storage_descriptors);
  TransferRequest request(agent, backend, operation, local, storage,
                          static_cast<size_t>(local_descriptors.descCount()));
  request.execute(stop);
}

std::vector<std::string> split_key(const std::string& key) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (start <= key.size()) {
    size_t separator = key.find(kKeySeparator, start);
    if (separator == std::string::npos) {
      fields.push_back(key.substr(start));
      break;
    }
    fields.push_back(key.substr(start, separator - start));
    start = separator + 1;
  }
  if (fields.size() != 4 && fields.size() != 5) {
    throw std::runtime_error(
        "malformed native key: expected four or five '@'-separated fields");
  }
  if (std::any_of(fields.begin(), fields.begin() + 4,
                  [](const std::string& field) { return field.empty(); })) {
    throw std::runtime_error("malformed native key: empty required field");
  }
  return fields;
}

bool parse_bool_parameter(
    const std::unordered_map<std::string, std::string>& parameters,
    const std::string& name, bool default_value = false) {
  auto iterator = parameters.find(name);
  if (iterator == parameters.end()) return default_value;
  if (iterator->second == "true") return true;
  if (iterator->second == "false") return false;
  throw std::runtime_error(name + " must be 'true' or 'false'");
}

std::vector<uint8_t> query_storage(nixlAgent& agent, nixlBackendH* backend,
                                   nixl_mem_t memory_type,
                                   const std::vector<std::string>& identities) {
  nixl_reg_dlist_t descriptors(memory_type);
  descriptors.reserve(identities.size());
  for (const std::string& identity : identities) {
    descriptors.addDesc(nixlBlobDesc(0, 0, 0, identity));
  }
  nixl_opt_args_t options;
  options.backends = {backend};
  std::vector<nixl_query_resp_t> responses;
  check_nixl(agent.queryMem(descriptors, responses, &options),
             "NIXL storage query");
  if (responses.size() != identities.size()) {
    throw std::runtime_error("NIXL query returned an invalid response count");
  }
  std::vector<uint8_t> results;
  results.reserve(responses.size());
  for (const nixl_query_resp_t& response : responses) {
    results.push_back(response.has_value() ? 1 : 0);
  }
  return results;
}

class NixlFileStorage final : public NixlStorageStrategy {
 public:
  NixlFileStorage(
      const std::unordered_map<std::string, std::string>& backend_params,
      size_t l1_alignment)
      : l1_alignment_(l1_alignment),
        use_direct_io_(
            parse_bool_parameter(backend_params, "use_direct_io", false)),
        shard_directories_(
            parse_bool_parameter(backend_params, "shard_dirs", false)) {
    auto path = backend_params.find("file_path");
    if (path == backend_params.end() || path->second.empty()) {
      throw std::runtime_error(
          "FILE storage requires backend_params.file_path");
    }
    base_path_ = std::filesystem::absolute(path->second).lexically_normal();
    std::filesystem::create_directories(base_path_);
    if (!std::filesystem::is_directory(base_path_)) {
      throw std::runtime_error("NIXL file_path is not a directory");
    }
    struct statvfs filesystem_info{};
    if (::statvfs(base_path_.c_str(), &filesystem_info) != 0) {
      throw std::runtime_error("statvfs failed for NIXL file_path: " +
                               std::string(std::strerror(errno)));
    }
    direct_io_alignment_ = static_cast<size_t>(filesystem_info.f_bsize);
  }

  void store(nixlAgent& agent, nixlBackendH* backend,
             const std::string& agent_name,
             const std::vector<NixlTransferBuffer>& buffers,
             const std::atomic<bool>& stop) override {
    nixl_reg_dlist_t registration(FILE_SEG);
    nixl_xfer_dlist_t local(DRAM_SEG);
    nixl_xfer_dlist_t storage(FILE_SEG);
    std::vector<TemporaryFile> temporary_files;
    std::vector<std::filesystem::path> final_paths;
    temporary_files.reserve(buffers.size());
    final_paths.reserve(buffers.size());

    for (size_t index = 0; index < buffers.size(); ++index) {
      const NixlTransferBuffer& buffer = buffers[index];
      validate_direct_io(buffer);
      std::filesystem::path final_path = path_for_key(buffer.key);
      std::filesystem::create_directories(final_path.parent_path());
      std::filesystem::path temporary_path = final_path;
      temporary_path += ".tmp." +
                        std::to_string(static_cast<long long>(getpid())) + "." +
                        std::to_string(next_temporary_id.fetch_add(1));
      int flags = O_CREAT | O_EXCL | O_RDWR;
#ifdef O_DIRECT
      if (use_direct_io_) flags |= O_DIRECT;
#endif
      int fd = ::open(temporary_path.c_str(), flags, 0644);
      if (fd < 0) {
        throw std::runtime_error("failed to create NIXL temporary file: " +
                                 std::string(std::strerror(errno)));
      }
      temporary_files.emplace_back(temporary_path, fd);
      final_paths.push_back(final_path);
      local.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer.data),
                                  buffer.length, 0));
      nixlBlobDesc file_descriptor(0, buffer.length, static_cast<uint64_t>(fd),
                                   std::string());
      registration.addDesc(file_descriptor);
      storage.addDesc(file_descriptor);
    }

    execute_transfer(agent, backend, agent_name, NIXL_WRITE, local,
                     registration, storage, stop);
    for (TemporaryFile& file : temporary_files) {
      if (::fsync(file.fd()) != 0) {
        throw std::runtime_error("fsync failed for NIXL temporary file: " +
                                 std::string(std::strerror(errno)));
      }
      file.close();
    }

    std::vector<std::filesystem::path> published;
    try {
      for (size_t index = 0; index < temporary_files.size(); ++index) {
        const std::filesystem::path& temporary_path =
            temporary_files[index].path();
        const std::filesystem::path& final_path = final_paths[index];
        if (::link(temporary_path.c_str(), final_path.c_str()) == 0) {
          published.push_back(final_path);
          std::filesystem::remove(temporary_path);
          continue;
        }
        if (errno != EEXIST) {
          throw std::runtime_error("atomic NIXL file publication failed: " +
                                   std::string(std::strerror(errno)));
        }
        struct stat existing{};
        if (::stat(final_path.c_str(), &existing) != 0 ||
            existing.st_size < 0 ||
            static_cast<size_t>(existing.st_size) != buffers[index].length) {
          throw std::runtime_error(
              "racing NIXL writer published a different file size");
        }
      }
    } catch (...) {
      for (const std::filesystem::path& path : published) {
        std::error_code error;
        std::filesystem::remove(path, error);
      }
      throw;
    }
  }

  std::vector<uint8_t> load(nixlAgent& agent, nixlBackendH* backend,
                            const std::string& agent_name,
                            const std::vector<NixlTransferBuffer>& buffers,
                            const std::atomic<bool>& stop) override {
    std::vector<uint8_t> results(buffers.size(), 0);
    nixl_reg_dlist_t registration(FILE_SEG);
    nixl_xfer_dlist_t local(DRAM_SEG);
    nixl_xfer_dlist_t storage(FILE_SEG);
    std::vector<UniqueFd> files;
    std::vector<size_t> selected_indices;

    for (size_t index = 0; index < buffers.size(); ++index) {
      const NixlTransferBuffer& buffer = buffers[index];
      validate_direct_io(buffer);
      std::filesystem::path path = path_for_key(buffer.key);
      int flags = O_RDONLY;
#ifdef O_DIRECT
      if (use_direct_io_) flags |= O_DIRECT;
#endif
      int fd = ::open(path.c_str(), flags);
      if (fd < 0) continue;
      struct stat file_info{};
      if (::fstat(fd, &file_info) != 0 || file_info.st_size < 0 ||
          static_cast<size_t>(file_info.st_size) != buffer.length) {
        ::close(fd);
        continue;
      }
      files.emplace_back(fd);
      selected_indices.push_back(index);
      local.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer.data),
                                  buffer.length, 0));
      nixlBlobDesc file_descriptor(0, buffer.length, static_cast<uint64_t>(fd),
                                   std::string());
      registration.addDesc(file_descriptor);
      storage.addDesc(file_descriptor);
    }

    try {
      execute_transfer(agent, backend, agent_name, NIXL_READ, local,
                       registration, storage, stop);
      for (size_t index : selected_indices) results[index] = 1;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "[LMCache NIXL GET] %s\n", error.what());
    }
    return results;
  }

  std::vector<uint8_t> exists(nixlAgent& agent, nixlBackendH* backend,
                              const std::vector<std::string>& keys) override {
    std::vector<std::string> paths;
    paths.reserve(keys.size());
    for (const std::string& key : keys) paths.push_back(path_for_key(key));
    return query_storage(agent, backend, FILE_SEG, paths);
  }

  std::vector<uint8_t> remove(const std::vector<std::string>& keys) override {
    std::vector<uint8_t> results;
    results.reserve(keys.size());
    for (const std::string& key : keys) {
      std::error_code error;
      bool removed = std::filesystem::remove(path_for_key(key), error);
      results.push_back(!error && removed ? 1 : 0);
    }
    return results;
  }

  NixlStorageCapabilities capabilities() const override {
    return {.query = true,
            .deletion = true,
            .direct_io = use_direct_io_,
            .atomic_publication = true};
  }

 private:
  std::filesystem::path path_for_key(const std::string& key) const {
    return base_path_ / nixl_persistent_identity(key, shard_directories_);
  }

  void validate_direct_io(const NixlTransferBuffer& buffer) const {
    if (!use_direct_io_) return;
    uintptr_t address = reinterpret_cast<uintptr_t>(buffer.data);
    size_t alignment = std::max(l1_alignment_, direct_io_alignment_);
    if (address % alignment != 0 || buffer.length % alignment != 0) {
      throw std::runtime_error(
          "direct I/O requires buffer address and length aligned to " +
          std::to_string(alignment) + " bytes");
    }
  }

  std::filesystem::path base_path_;
  size_t l1_alignment_;
  size_t direct_io_alignment_ = 0;
  bool use_direct_io_;
  bool shard_directories_;
};

class NixlObjectStorage final : public NixlStorageStrategy {
 public:
  explicit NixlObjectStorage(
      const std::unordered_map<std::string, std::string>& backend_params)
      : shard_directories_(
            parse_bool_parameter(backend_params, "shard_dirs", false)) {}

  void store(nixlAgent& agent, nixlBackendH* backend,
             const std::string& agent_name,
             const std::vector<NixlTransferBuffer>& buffers,
             const std::atomic<bool>& stop) override {
    nixl_reg_dlist_t registration(OBJ_SEG);
    nixl_xfer_dlist_t local(DRAM_SEG);
    nixl_xfer_dlist_t storage(OBJ_SEG);
    for (size_t index = 0; index < buffers.size(); ++index) {
      const NixlTransferBuffer& buffer = buffers[index];
      local.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer.data),
                                  buffer.length, 0));
      nixlBlobDesc object_descriptor(
          0, buffer.length, index,
          nixl_persistent_identity(buffer.key, shard_directories_));
      registration.addDesc(object_descriptor);
      storage.addDesc(object_descriptor);
    }
    execute_transfer(agent, backend, agent_name, NIXL_WRITE, local,
                     registration, storage, stop);
  }

  std::vector<uint8_t> load(nixlAgent& agent, nixlBackendH* backend,
                            const std::string& agent_name,
                            const std::vector<NixlTransferBuffer>& buffers,
                            const std::atomic<bool>& stop) override {
    std::vector<std::string> identities;
    identities.reserve(buffers.size());
    for (const NixlTransferBuffer& buffer : buffers) {
      identities.push_back(
          nixl_persistent_identity(buffer.key, shard_directories_));
    }
    std::vector<uint8_t> results =
        query_storage(agent, backend, OBJ_SEG, identities);
    nixl_reg_dlist_t registration(OBJ_SEG);
    nixl_xfer_dlist_t local(DRAM_SEG);
    nixl_xfer_dlist_t storage(OBJ_SEG);
    std::vector<size_t> selected_indices;
    for (size_t index = 0; index < buffers.size(); ++index) {
      if (!results[index]) continue;
      results[index] = 0;
      selected_indices.push_back(index);
      const NixlTransferBuffer& buffer = buffers[index];
      local.addDesc(nixlBasicDesc(reinterpret_cast<uintptr_t>(buffer.data),
                                  buffer.length, 0));
      nixlBlobDesc object_descriptor(0, buffer.length, index,
                                     identities[index]);
      registration.addDesc(object_descriptor);
      storage.addDesc(object_descriptor);
    }
    try {
      execute_transfer(agent, backend, agent_name, NIXL_READ, local,
                       registration, storage, stop);
      for (size_t index : selected_indices) results[index] = 1;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "[LMCache NIXL GET] %s\n", error.what());
    }
    return results;
  }

  std::vector<uint8_t> exists(nixlAgent& agent, nixlBackendH* backend,
                              const std::vector<std::string>& keys) override {
    std::vector<std::string> identities;
    identities.reserve(keys.size());
    for (const std::string& key : keys) {
      identities.push_back(nixl_persistent_identity(key, shard_directories_));
    }
    return query_storage(agent, backend, OBJ_SEG, identities);
  }

  std::vector<uint8_t> remove(const std::vector<std::string>& keys) override {
    (void)keys;
    throw std::runtime_error(
        "NIXL 1.3 OBJECT storage does not support deletion");
  }

  NixlStorageCapabilities capabilities() const override {
    return {.query = true,
            .deletion = false,
            .direct_io = false,
            .atomic_publication = false};
  }

 private:
  bool shard_directories_;
};

}  // namespace

std::string nixl_persistent_identity(const std::string& serialized_key,
                                     bool shard_directories) {
  std::vector<std::string> fields = split_key(serialized_key);
  std::string safe_model = fields[0];
  size_t slash = 0;
  while ((slash = safe_model.find('/', slash)) != std::string::npos) {
    safe_model.replace(slash, 1, "--");
    slash += 2;
  }
  std::string filename =
      safe_model + "_" + fields[1] + "_" + fields[2] + "_" + fields[3];
  if (fields.size() == 5 && !fields[4].empty()) {
    filename += "@" + fields[4];
  }
  filename += kFileExtension;
  if (filename.size() > 255) {
    throw std::runtime_error(
        "native NIXL persistent filename exceeds 255 bytes");
  }
  if (!shard_directories) return filename;
  if (fields[3].size() < 4) {
    throw std::runtime_error(
        "shard_dirs requires a chunk hash of at least two bytes");
  }
  return fields[3].substr(0, 2) + "/" + fields[3].substr(2, 2) + "/" + filename;
}

std::unique_ptr<NixlStorageStrategy> make_nixl_storage_strategy(
    NixlStorageKind storage_kind,
    const std::unordered_map<std::string, std::string>& backend_params,
    size_t l1_alignment) {
  switch (storage_kind) {
    case NixlStorageKind::File:
      return std::make_unique<NixlFileStorage>(backend_params, l1_alignment);
    case NixlStorageKind::Object:
      return std::make_unique<NixlObjectStorage>(backend_params);
  }
  throw std::runtime_error("unsupported inferred NIXL storage kind");
}

const char* nixl_storage_kind_name(NixlStorageKind storage_kind) {
  switch (storage_kind) {
    case NixlStorageKind::File:
      return "FILE";
    case NixlStorageKind::Object:
      return "OBJECT";
  }
  throw std::runtime_error("unsupported inferred NIXL storage kind");
}

}  // namespace connector
}  // namespace lmcache
