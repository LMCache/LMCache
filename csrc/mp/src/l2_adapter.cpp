// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/l2_adapter.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

namespace lmcache::mp {
namespace {

namespace fs = std::filesystem;

void SetError(std::string* error, std::string message) {
  if (error != nullptr) {
    *error = std::move(message);
  }
}

std::optional<std::string> JsonStringField(const std::string& json,
                                           const std::string& field) {
  const std::string marker = "\"" + field + "\"";
  std::size_t pos = json.find(marker);
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  pos = json.find(':', pos + marker.size());
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  pos = json.find('"', pos + 1);
  if (pos == std::string::npos) {
    return std::nullopt;
  }
  std::string out;
  bool escaped = false;
  for (std::size_t i = pos + 1; i < json.size(); ++i) {
    const char ch = json[i];
    if (escaped) {
      out.push_back(ch);
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '"') {
      return out;
    }
    out.push_back(ch);
  }
  return std::nullopt;
}

std::string ReplaceSlash(std::string value) {
  static const std::string kNeedle = "/";
  static const std::string kReplacement = "-SEP-";
  std::size_t pos = 0;
  while ((pos = value.find(kNeedle, pos)) != std::string::npos) {
    value.replace(pos, kNeedle.size(), kReplacement);
    pos += kReplacement.size();
  }
  return value;
}

std::string CacheSaltFromFsFilename(std::string filename) {
  static constexpr char kDataSuffix[] = ".data";
  if (filename.size() >= std::strlen(kDataSuffix) &&
      filename.compare(filename.size() - std::strlen(kDataSuffix),
                       std::strlen(kDataSuffix), kDataSuffix) == 0) {
    filename.resize(filename.size() - std::strlen(kDataSuffix));
  }

  const std::size_t first = filename.find('@');
  if (first == std::string::npos) {
    return "";
  }
  const std::size_t second = filename.find('@', first + 1);
  if (second == std::string::npos) {
    return "";
  }
  const std::size_t third = filename.find('@', second + 1);
  if (third == std::string::npos) {
    return "";
  }
  return filename.substr(third + 1);
}

class FileSystemL2Adapter final : public L2Adapter {
 public:
  explicit FileSystemL2Adapter(fs::path base_path)
      : base_path_(std::move(base_path)) {
    fs::create_directories(base_path_);
  }

  bool Put(const std::string& object_key,
           const std::vector<std::uint8_t>& bytes,
           std::string* error) override {
    const fs::path path = path_for_key(object_key);
    const fs::path tmp = path.string() + ".tmp";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      SetError(error, "cannot open " + tmp.string() + " for write");
      return false;
    }
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (!out) {
      SetError(error, "cannot write " + tmp.string());
      return false;
    }
    out.close();
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
      SetError(error, "cannot rename " + tmp.string() + " to " + path.string() +
                          ": " + ec.message());
      return false;
    }
    return true;
  }

  std::optional<std::vector<std::uint8_t>> Get(const std::string& object_key,
                                               std::string* error) override {
    const fs::path path = path_for_key(object_key);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return std::nullopt;
    }
    in.seekg(0, std::ios::end);
    const auto size = in.tellg();
    if (size < 0) {
      SetError(error, "cannot stat " + path.string());
      return std::nullopt;
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char*>(bytes.data()), size);
    if (!in && size != 0) {
      SetError(error, "cannot read " + path.string());
      return std::nullopt;
    }
    return bytes;
  }

  bool Delete(const std::string& object_key, std::string* error) override {
    std::error_code ec;
    fs::remove(path_for_key(object_key), ec);
    if (ec) {
      SetError(error, ec.message());
      return false;
    }
    return true;
  }

  bool Clear(std::string* error) override {
    std::error_code ec;
    if (!fs::exists(base_path_, ec)) {
      if (ec) {
        SetError(error, ec.message());
        return false;
      }
      return true;
    }

    for (const fs::directory_entry& entry :
         fs::directory_iterator(base_path_, ec)) {
      if (ec) {
        SetError(error, ec.message());
        return false;
      }
      std::error_code entry_ec;
      if (!entry.is_regular_file(entry_ec)) {
        if (entry_ec) {
          SetError(error, entry_ec.message());
          return false;
        }
        continue;
      }
      fs::remove(entry.path(), entry_ec);
      if (entry_ec) {
        SetError(error, entry_ec.message());
        return false;
      }
    }
    return true;
  }

  bool Exists(const std::string& object_key) const override {
    return fs::exists(path_for_key(object_key));
  }

  L2AdapterStatus Status() const override {
    L2AdapterStatus status{
        .type = "fs",
        .base_path = base_path_.string(),
    };
    std::error_code ec;
    if (!fs::exists(base_path_, ec)) {
      return status;
    }
    for (const fs::directory_entry& entry :
         fs::directory_iterator(base_path_, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_regular_file()) {
        continue;
      }
      ++status.stored_files;
      status.stored_bytes += entry.file_size(ec);
      if (ec) {
        ec.clear();
      }
    }
    return status;
  }

  std::unordered_map<std::string, std::uint64_t> UsageBytesByCacheSalt()
      const override {
    std::unordered_map<std::string, std::uint64_t> usage;
    std::error_code ec;
    if (!fs::exists(base_path_, ec)) {
      return usage;
    }
    for (const fs::directory_entry& entry :
         fs::directory_iterator(base_path_, ec)) {
      if (ec) {
        break;
      }
      std::error_code entry_ec;
      if (!entry.is_regular_file(entry_ec) || entry_ec) {
        continue;
      }
      const std::uint64_t bytes = entry.file_size(entry_ec);
      if (entry_ec) {
        continue;
      }
      const std::string salt =
          CacheSaltFromFsFilename(entry.path().filename().string());
      usage[salt] += bytes;
    }
    return usage;
  }

 private:
  fs::path path_for_key(const std::string& object_key) const {
    return base_path_ / ObjectKeyStringToFsFilename(object_key);
  }

  fs::path base_path_;
};

}  // namespace

std::string ObjectKeyStringToFsFilename(const std::string& object_key) {
  const std::size_t first = object_key.find('@');
  if (first == std::string::npos) {
    return ReplaceSlash(object_key) + ".data";
  }
  const std::size_t second = object_key.find('@', first + 1);
  if (second == std::string::npos) {
    return ReplaceSlash(object_key) + ".data";
  }

  const std::string safe_model = ReplaceSlash(object_key.substr(0, first));
  const std::string rank = object_key.substr(first + 1, second - first - 1);
  const std::string rest = object_key.substr(second + 1);
  std::ostringstream out;
  out << safe_model << "@0x" << rank << "@" << rest << ".data";
  return out.str();
}

std::optional<L2AdapterConfig> ParseL2AdapterConfig(const std::string& json,
                                                    std::string* error) {
  const auto type = JsonStringField(json, "type");
  if (!type) {
    SetError(error, "L2 adapter config must contain string field 'type'");
    return std::nullopt;
  }
  if (*type == "nixl") {
    SetError(error,
             "native MP does not implement the NIXL L2 adapter yet; use the "
             "Python MP server for NIXL or configure native type 'fs'");
    return std::nullopt;
  }
  if (*type != "fs") {
    SetError(error, "native MP only supports L2 adapter type 'fs'; got '" +
                        *type + "'");
    return std::nullopt;
  }

  const auto base_path = JsonStringField(json, "base_path");
  if (!base_path || base_path->empty()) {
    SetError(error,
             "fs L2 adapter requires non-empty string field 'base_path'");
    return std::nullopt;
  }
  if (error != nullptr) {
    error->clear();
  }
  return L2AdapterConfig{.type = *type, .base_path = *base_path};
}

std::unique_ptr<L2Adapter> CreateL2Adapter(const L2AdapterConfig& config,
                                           std::string* error) {
  if (config.type != "fs") {
    SetError(error, "unsupported native L2 adapter type '" + config.type + "'");
    return nullptr;
  }
  try {
    auto adapter = std::make_unique<FileSystemL2Adapter>(config.base_path);
    if (error != nullptr) {
      error->clear();
    }
    return adapter;
  } catch (const std::exception& exc) {
    SetError(error, exc.what());
    return nullptr;
  }
}

}  // namespace lmcache::mp

struct LmcacheMpCppL2Adapter {
  std::unique_ptr<lmcache::mp::L2Adapter> impl;
  std::string last_error;
};

extern "C" {

LmcacheMpCppL2Adapter* lmcache_mp_cpp_fs_l2_create(const char* base_path) {
  if (base_path == nullptr) {
    return nullptr;
  }
  std::string error;
  auto adapter = lmcache::mp::CreateL2Adapter(
      {.type = "fs", .base_path = base_path}, &error);
  if (!adapter) {
    return nullptr;
  }
  return new LmcacheMpCppL2Adapter{.impl = std::move(adapter)};
}

void lmcache_mp_cpp_fs_l2_destroy(LmcacheMpCppL2Adapter* adapter) {
  delete adapter;
}

int lmcache_mp_cpp_fs_l2_put(LmcacheMpCppL2Adapter* adapter, const char* key,
                             const std::uint8_t* data, std::uint64_t len) {
  if (adapter == nullptr || key == nullptr || data == nullptr) {
    return -1;
  }
  adapter->last_error.clear();
  return adapter->impl->Put(key, std::vector<std::uint8_t>(data, data + len),
                            &adapter->last_error)
             ? 1
             : -1;
}

int lmcache_mp_cpp_fs_l2_size(LmcacheMpCppL2Adapter* adapter, const char* key,
                              std::uint64_t* out_len) {
  if (adapter == nullptr || key == nullptr || out_len == nullptr) {
    return -1;
  }
  adapter->last_error.clear();
  auto bytes = adapter->impl->Get(key, &adapter->last_error);
  if (!bytes) {
    return 0;
  }
  *out_len = bytes->size();
  return 1;
}

int lmcache_mp_cpp_fs_l2_get(LmcacheMpCppL2Adapter* adapter, const char* key,
                             std::uint8_t* out, std::uint64_t len) {
  if (adapter == nullptr || key == nullptr || out == nullptr) {
    return -1;
  }
  adapter->last_error.clear();
  auto bytes = adapter->impl->Get(key, &adapter->last_error);
  if (!bytes) {
    return 0;
  }
  if (bytes->size() != len) {
    adapter->last_error = "size mismatch for L2 key";
    return -2;
  }
  std::memcpy(out, bytes->data(), bytes->size());
  return 1;
}

int lmcache_mp_cpp_fs_l2_delete(LmcacheMpCppL2Adapter* adapter,
                                const char* key) {
  if (adapter == nullptr || key == nullptr) {
    return -1;
  }
  adapter->last_error.clear();
  return adapter->impl->Delete(key, &adapter->last_error) ? 1 : -1;
}

int lmcache_mp_cpp_fs_l2_clear(LmcacheMpCppL2Adapter* adapter) {
  if (adapter == nullptr) {
    return -1;
  }
  adapter->last_error.clear();
  return adapter->impl->Clear(&adapter->last_error) ? 1 : -1;
}

int lmcache_mp_cpp_fs_l2_exists(LmcacheMpCppL2Adapter* adapter,
                                const char* key) {
  if (adapter == nullptr || key == nullptr) {
    return -1;
  }
  return adapter->impl->Exists(key) ? 1 : 0;
}

const char* lmcache_mp_cpp_fs_l2_last_error(LmcacheMpCppL2Adapter* adapter) {
  if (adapter == nullptr) {
    return "adapter is null";
  }
  return adapter->last_error.c_str();
}

int lmcache_mp_cpp_fs_l2_filename(const char* key, char* out,
                                  std::uint64_t out_len,
                                  std::uint64_t* needed_len) {
  if (key == nullptr || out == nullptr || needed_len == nullptr) {
    return -1;
  }
  const std::string filename = lmcache::mp::ObjectKeyStringToFsFilename(key);
  *needed_len = filename.size();
  if (out_len <= filename.size()) {
    return -2;
  }
  std::memcpy(out, filename.data(), filename.size());
  out[filename.size()] = '\0';
  return 1;
}
}
