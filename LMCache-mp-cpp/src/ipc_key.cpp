// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/ipc_key.h"

#include "lmcache_mp_cpp/key_compat.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace lmcache::mp {
namespace {

class Reader {
 public:
  Reader(const std::uint8_t* data, std::uint64_t len)
      : data_(data), len_(static_cast<std::size_t>(len)) {}

  bool Eof() const { return pos_ == len_; }

  bool ReadMapLen(std::uint64_t* len) {
    return ReadContainerLen(len, 0x80, 0x8f, 0xde, 0xdf);
  }

  bool ReadArrayLen(std::uint64_t* len) {
    return ReadContainerLen(len, 0x90, 0x9f, 0xdc, 0xdd);
  }

  bool ReadString(std::string* out) {
    if (!CanRead(1)) {
      return false;
    }
    const std::uint8_t tag = data_[pos_++];
    std::uint64_t len = 0;
    if ((tag & 0xe0) == 0xa0) {
      len = tag & 0x1f;
    } else if (tag == 0xd9) {
      if (!ReadU8(&len)) {
        return false;
      }
    } else if (tag == 0xda) {
      if (!ReadU16(&len)) {
        return false;
      }
    } else if (tag == 0xdb) {
      if (!ReadU32(&len)) {
        return false;
      }
    } else {
      return false;
    }
    if (len > std::numeric_limits<std::size_t>::max() ||
        !CanRead(static_cast<std::size_t>(len))) {
      return false;
    }
    out->assign(reinterpret_cast<const char*>(data_ + pos_),
                static_cast<std::size_t>(len));
    pos_ += static_cast<std::size_t>(len);
    return true;
  }

  bool ReadUnsigned(std::uint64_t* out) {
    if (!CanRead(1)) {
      return false;
    }
    const std::uint8_t tag = data_[pos_++];
    if (tag <= 0x7f) {
      *out = tag;
      return true;
    }
    switch (tag) {
      case 0xcc:
        return ReadU8(out);
      case 0xcd:
        return ReadU16(out);
      case 0xce:
        return ReadU32(out);
      case 0xcf:
        return ReadU64(out);
      case 0xd0: {
        std::uint64_t raw = 0;
        if (!ReadU8(&raw) || (raw & 0x80) != 0) {
          return false;
        }
        *out = raw;
        return true;
      }
      case 0xd1: {
        std::uint64_t raw = 0;
        if (!ReadU16(&raw) || (raw & 0x8000) != 0) {
          return false;
        }
        *out = raw;
        return true;
      }
      case 0xd2: {
        std::uint64_t raw = 0;
        if (!ReadU32(&raw) || (raw & 0x80000000ULL) != 0) {
          return false;
        }
        *out = raw;
        return true;
      }
      case 0xd3: {
        std::uint64_t raw = 0;
        if (!ReadU64(&raw) || (raw & 0x8000000000000000ULL) != 0) {
          return false;
        }
        *out = raw;
        return true;
      }
      default:
        return false;
    }
  }

  bool TryReadNil(bool* is_nil) {
    if (!CanRead(1)) {
      return false;
    }
    if (data_[pos_] == 0xc0) {
      ++pos_;
      *is_nil = true;
      return true;
    }
    *is_nil = false;
    return true;
  }

  bool SkipValue() {
    if (!CanRead(1)) {
      return false;
    }
    const std::uint8_t tag = data_[pos_++];
    if (tag <= 0x7f || tag >= 0xe0 || tag == 0xc0 || tag == 0xc2 ||
        tag == 0xc3) {
      return true;
    }
    if ((tag & 0xe0) == 0xa0) {
      return SkipBytes(tag & 0x1f);
    }
    if ((tag & 0xf0) == 0x90) {
      return SkipN(tag & 0x0f);
    }
    if ((tag & 0xf0) == 0x80) {
      return SkipN(2 * (tag & 0x0f));
    }

    std::uint64_t len = 0;
    switch (tag) {
      case 0xc4:
      case 0xd9:
        return ReadU8(&len) && SkipBytes(len);
      case 0xc5:
      case 0xda:
        return ReadU16(&len) && SkipBytes(len);
      case 0xc6:
      case 0xdb:
        return ReadU32(&len) && SkipBytes(len);
      case 0xdc:
        return ReadU16(&len) && SkipN(len);
      case 0xdd:
        return ReadU32(&len) && SkipN(len);
      case 0xde:
        return ReadU16(&len) && SkipN(2 * len);
      case 0xdf:
        return ReadU32(&len) && SkipN(2 * len);
      case 0xca:
        return SkipBytes(4);
      case 0xcb:
        return SkipBytes(8);
      case 0xcc:
      case 0xd0:
        return SkipBytes(1);
      case 0xcd:
      case 0xd1:
        return SkipBytes(2);
      case 0xce:
      case 0xd2:
        return SkipBytes(4);
      case 0xcf:
      case 0xd3:
        return SkipBytes(8);
      case 0xd4:
        return SkipBytes(2);
      case 0xd5:
        return SkipBytes(3);
      case 0xd6:
        return SkipBytes(5);
      case 0xd7:
        return SkipBytes(9);
      case 0xd8:
        return SkipBytes(17);
      case 0xc7:
        return ReadU8(&len) && SkipBytes(1 + len);
      case 0xc8:
        return ReadU16(&len) && SkipBytes(1 + len);
      case 0xc9:
        return ReadU32(&len) && SkipBytes(1 + len);
      default:
        return false;
    }
  }

 private:
  bool ReadContainerLen(std::uint64_t* len, std::uint8_t fix_start,
                        std::uint8_t fix_end, std::uint8_t tag16,
                        std::uint8_t tag32) {
    if (!CanRead(1)) {
      return false;
    }
    const std::uint8_t tag = data_[pos_++];
    if (tag >= fix_start && tag <= fix_end) {
      *len = tag & 0x0f;
      return true;
    }
    if (tag == tag16) {
      return ReadU16(len);
    }
    if (tag == tag32) {
      return ReadU32(len);
    }
    return false;
  }

  bool ReadU8(std::uint64_t* out) {
    if (!CanRead(1)) {
      return false;
    }
    *out = data_[pos_++];
    return true;
  }

  bool ReadU16(std::uint64_t* out) {
    if (!CanRead(2)) {
      return false;
    }
    *out = (static_cast<std::uint64_t>(data_[pos_]) << 8) |
           static_cast<std::uint64_t>(data_[pos_ + 1]);
    pos_ += 2;
    return true;
  }

  bool ReadU32(std::uint64_t* out) {
    if (!CanRead(4)) {
      return false;
    }
    *out = (static_cast<std::uint64_t>(data_[pos_]) << 24) |
           (static_cast<std::uint64_t>(data_[pos_ + 1]) << 16) |
           (static_cast<std::uint64_t>(data_[pos_ + 2]) << 8) |
           static_cast<std::uint64_t>(data_[pos_ + 3]);
    pos_ += 4;
    return true;
  }

  bool ReadU64(std::uint64_t* out) {
    if (!CanRead(8)) {
      return false;
    }
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value = (value << 8) | data_[pos_ + static_cast<std::size_t>(i)];
    }
    pos_ += 8;
    *out = value;
    return true;
  }

  bool SkipN(std::uint64_t count) {
    for (std::uint64_t i = 0; i < count; ++i) {
      if (!SkipValue()) {
        return false;
      }
    }
    return true;
  }

  bool SkipBytes(std::uint64_t len) {
    if (len > std::numeric_limits<std::size_t>::max() ||
        !CanRead(static_cast<std::size_t>(len))) {
      return false;
    }
    pos_ += static_cast<std::size_t>(len);
    return true;
  }

  bool CanRead(std::size_t n) const { return n <= len_ - pos_; }

  const std::uint8_t* data_;
  std::size_t len_;
  std::size_t pos_ = 0;
};

void SetError(std::string* error, const char* message) {
  if (error != nullptr) {
    *error = message;
  }
}

bool ReadTokenArray(Reader* reader, std::vector<std::uint32_t>* tokens) {
  std::uint64_t len = 0;
  if (!reader->ReadArrayLen(&len)) {
    return false;
  }
  if (len >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  tokens->clear();
  tokens->reserve(static_cast<std::size_t>(len));
  for (std::uint64_t i = 0; i < len; ++i) {
    std::uint64_t token = 0;
    if (!reader->ReadUnsigned(&token) ||
        token > std::numeric_limits<std::uint32_t>::max()) {
      return false;
    }
    tokens->push_back(static_cast<std::uint32_t>(token));
  }
  return true;
}

bool ValidateCacheSalt(const std::string& cache_salt) {
  return cache_salt.size() <= 128 &&
         cache_salt.find_first_of("@/\\\0", 0, 4) == std::string::npos;
}

LmcacheMpCppIpcKeySummary SummaryFromKey(const IpcCacheEngineKey& key) {
  return {
      static_cast<std::uint64_t>(key.token_ids.size()),
      key.start,
      key.end,
      static_cast<std::uint64_t>(key.model_name.size()),
      static_cast<std::uint64_t>(key.request_id.size()),
      static_cast<std::uint64_t>(key.cache_salt.size()),
      key.world_size,
      key.worker_id ? static_cast<std::int32_t>(*key.worker_id) : -1,
  };
}

bool CopyString(const std::string& value, char* out, std::uint64_t out_len) {
  if (out == nullptr || out_len <= value.size()) {
    return false;
  }
  std::memcpy(out, value.data(), value.size());
  out[value.size()] = '\0';
  return true;
}

}  // namespace

std::optional<IpcCacheEngineKey> DecodeIpcCacheEngineKey(
    const std::uint8_t* data, std::uint64_t len, std::string* error) {
  if (data == nullptr) {
    SetError(error, "null payload");
    return std::nullopt;
  }

  Reader reader(data, len);
  std::uint64_t map_len = 0;
  if (!reader.ReadMapLen(&map_len)) {
    SetError(error, "IPCCacheEngineKey payload is not a msgpack map");
    return std::nullopt;
  }

  IpcCacheEngineKey key;
  bool has_model_name = false;
  bool has_world_size = false;
  bool has_worker_id = false;
  bool has_token_ids = false;
  bool has_start = false;
  bool has_end = false;
  bool has_request_id = false;

  for (std::uint64_t i = 0; i < map_len; ++i) {
    std::string field;
    if (!reader.ReadString(&field)) {
      SetError(error, "IPCCacheEngineKey field name is not a string");
      return std::nullopt;
    }

    if (field == "model_name") {
      has_model_name = reader.ReadString(&key.model_name);
      if (!has_model_name) {
        SetError(error, "invalid model_name field");
        return std::nullopt;
      }
    } else if (field == "world_size") {
      std::uint64_t value = 0;
      if (!reader.ReadUnsigned(&value) ||
          value > std::numeric_limits<std::uint32_t>::max()) {
        SetError(error, "invalid world_size field");
        return std::nullopt;
      }
      key.world_size = static_cast<std::uint32_t>(value);
      has_world_size = true;
    } else if (field == "worker_id") {
      bool is_nil = false;
      if (!reader.TryReadNil(&is_nil)) {
        SetError(error, "invalid worker_id field");
        return std::nullopt;
      }
      if (is_nil) {
        key.worker_id = std::nullopt;
      } else {
        std::uint64_t value = 0;
        if (!reader.ReadUnsigned(&value) ||
            value > std::numeric_limits<std::uint32_t>::max()) {
          SetError(error, "invalid worker_id field");
          return std::nullopt;
        }
        key.worker_id = static_cast<std::uint32_t>(value);
      }
      has_worker_id = true;
    } else if (field == "token_ids") {
      if (!ReadTokenArray(&reader, &key.token_ids)) {
        SetError(error, "invalid token_ids field");
        return std::nullopt;
      }
      has_token_ids = true;
    } else if (field == "start") {
      if (!reader.ReadUnsigned(&key.start)) {
        SetError(error, "invalid start field");
        return std::nullopt;
      }
      has_start = true;
    } else if (field == "end") {
      if (!reader.ReadUnsigned(&key.end)) {
        SetError(error, "invalid end field");
        return std::nullopt;
      }
      has_end = true;
    } else if (field == "request_id") {
      has_request_id = reader.ReadString(&key.request_id);
      if (!has_request_id) {
        SetError(error, "invalid request_id field");
        return std::nullopt;
      }
    } else if (field == "cache_salt") {
      if (!reader.ReadString(&key.cache_salt)) {
        SetError(error, "invalid cache_salt field");
        return std::nullopt;
      }
    } else if (!reader.SkipValue()) {
      SetError(error, "invalid unknown IPCCacheEngineKey field value");
      return std::nullopt;
    }
  }

  if (!reader.Eof()) {
    SetError(error, "trailing bytes after IPCCacheEngineKey payload");
    return std::nullopt;
  }
  if (!has_model_name || !has_world_size || !has_worker_id || !has_token_ids ||
      !has_start || !has_end || !has_request_id) {
    SetError(error, "missing required IPCCacheEngineKey field");
    return std::nullopt;
  }
  if (key.world_size == 0) {
    SetError(error, "world_size must be positive");
    return std::nullopt;
  }
  if (key.worker_id && *key.worker_id >= key.world_size) {
    SetError(error, "worker_id must be less than world_size");
    return std::nullopt;
  }
  if (!ValidateCacheSalt(key.cache_salt)) {
    SetError(error, "invalid cache_salt field");
    return std::nullopt;
  }
  if (error != nullptr) {
    error->clear();
  }
  return key;
}

std::vector<std::string> ObjectKeyStringsForIpcKey(const IpcCacheEngineKey& key,
                                                   std::uint64_t chunk_size,
                                                   std::uint64_t start,
                                                   std::uint64_t end,
                                                   std::string* error) {
  if (chunk_size == 0) {
    SetError(error, "chunk_size must be positive");
    return {};
  }
  if (start > end || start % chunk_size != 0 || end % chunk_size != 0) {
    SetError(error, "start/end must be chunk aligned");
    return {};
  }

  const std::uint64_t effective_end =
      std::min<std::uint64_t>(end, key.token_ids.size());
  if (key.token_ids.empty() || effective_end <= start) {
    if (error != nullptr) {
      error->clear();
    }
    return {};
  }

  const std::uint64_t max_hashes = (effective_end - start) / chunk_size;
  if (max_hashes == 0) {
    if (error != nullptr) {
      error->clear();
    }
    return {};
  }

  std::vector<std::uint8_t> hashes(max_hashes * LMCACHE_MP_CPP_BLAKE3_OUT_LEN);
  std::uint64_t out_hashes = 0;
  const int hash_rc = lmcache_mp_cpp_blake3_chunk_hashes(
      key.token_ids.data(), key.token_ids.size(), chunk_size, start,
      effective_end, hashes.data(), max_hashes, &out_hashes);
  if (hash_rc != 1) {
    SetError(error, "failed to compute chunk hashes");
    return {};
  }

  const std::uint64_t max_ranks = key.worker_id ? 1 : key.world_size;
  std::vector<std::uint32_t> ranks(max_ranks);
  std::uint64_t rank_count = 0;
  const int rank_rc = lmcache_mp_cpp_expand_kv_ranks(
      key.world_size,
      key.worker_id ? static_cast<std::int32_t>(*key.worker_id) : -1,
      ranks.data(), ranks.size(), &rank_count);
  if (rank_rc != 1) {
    SetError(error, "failed to expand KV ranks");
    return {};
  }

  std::vector<std::string> object_keys;
  object_keys.reserve(static_cast<std::size_t>(out_hashes * rank_count));
  for (std::uint64_t i = 0; i < out_hashes; ++i) {
    const std::uint8_t* chunk_hash =
        hashes.data() + i * LMCACHE_MP_CPP_BLAKE3_OUT_LEN;
    for (std::uint64_t j = 0; j < rank_count; ++j) {
      std::uint64_t needed = 0;
      std::string encoded(key.model_name.size() +
                              LMCACHE_MP_CPP_BLAKE3_OUT_LEN * 2 +
                              key.cache_salt.size() + 32,
                          '\0');
      int rc = lmcache_mp_cpp_object_key_string(
          key.model_name.c_str(), ranks[j], chunk_hash,
          LMCACHE_MP_CPP_BLAKE3_OUT_LEN, key.cache_salt.c_str(), encoded.data(),
          encoded.size(), &needed);
      if (rc == -2) {
        encoded.assign(static_cast<std::size_t>(needed + 1), '\0');
        rc = lmcache_mp_cpp_object_key_string(
            key.model_name.c_str(), ranks[j], chunk_hash,
            LMCACHE_MP_CPP_BLAKE3_OUT_LEN, key.cache_salt.c_str(),
            encoded.data(), encoded.size(), &needed);
      }
      if (rc != 1) {
        SetError(error, "failed to encode ObjectKey");
        return {};
      }
      encoded.resize(static_cast<std::size_t>(needed));
      object_keys.push_back(std::move(encoded));
    }
  }

  if (error != nullptr) {
    error->clear();
  }
  return object_keys;
}

std::vector<std::string> LookupObjectKeyStringsForIpcKey(
    const IpcCacheEngineKey& key, std::uint64_t chunk_size,
    std::string* error) {
  const std::uint64_t end =
      (static_cast<std::uint64_t>(key.token_ids.size()) / chunk_size) *
      chunk_size;
  return ObjectKeyStringsForIpcKey(key, chunk_size, 0, end, error);
}

}  // namespace lmcache::mp

extern "C" {

int lmcache_mp_cpp_ipc_key_summary(const std::uint8_t* data, std::uint64_t len,
                                   LmcacheMpCppIpcKeySummary* out) {
  if (out == nullptr) {
    return -1;
  }
  std::string error;
  auto key = lmcache::mp::DecodeIpcCacheEngineKey(data, len, &error);
  if (!key) {
    return 0;
  }
  *out = lmcache::mp::SummaryFromKey(*key);
  return 1;
}

int lmcache_mp_cpp_decode_ipc_key(
    const std::uint8_t* data, std::uint64_t len, LmcacheMpCppIpcKeySummary* out,
    char* model_name, std::uint64_t model_name_len, char* request_id,
    std::uint64_t request_id_len, char* cache_salt,
    std::uint64_t cache_salt_len, std::uint32_t* token_ids,
    std::uint64_t max_token_ids) {
  if (out == nullptr) {
    return -1;
  }
  std::string error;
  auto key = lmcache::mp::DecodeIpcCacheEngineKey(data, len, &error);
  if (!key) {
    return 0;
  }
  *out = lmcache::mp::SummaryFromKey(*key);
  if (max_token_ids < key->token_ids.size() ||
      (!key->token_ids.empty() && token_ids == nullptr) ||
      !lmcache::mp::CopyString(key->model_name, model_name, model_name_len) ||
      !lmcache::mp::CopyString(key->request_id, request_id, request_id_len) ||
      !lmcache::mp::CopyString(key->cache_salt, cache_salt, cache_salt_len)) {
    return -2;
  }
  if (!key->token_ids.empty()) {
    std::memcpy(token_ids, key->token_ids.data(),
                key->token_ids.size() * sizeof(std::uint32_t));
  }
  return 1;
}

int lmcache_mp_cpp_ipc_key_object_key_strings(
    const std::uint8_t* data, std::uint64_t len, std::uint64_t chunk_size,
    std::uint64_t start, std::uint64_t end, char* out, std::uint64_t out_len,
    std::uint64_t* needed_len, std::uint64_t* out_count) {
  if (needed_len == nullptr || out_count == nullptr) {
    return -1;
  }
  std::string error;
  auto key = lmcache::mp::DecodeIpcCacheEngineKey(data, len, &error);
  if (!key) {
    return 0;
  }
  std::vector<std::string> object_keys = lmcache::mp::ObjectKeyStringsForIpcKey(
      *key, chunk_size, start, end, &error);
  if (!error.empty()) {
    return -3;
  }

  std::uint64_t needed = 0;
  for (const std::string& object_key : object_keys) {
    needed += static_cast<std::uint64_t>(object_key.size()) + 1;
  }
  *needed_len = needed;
  *out_count = static_cast<std::uint64_t>(object_keys.size());
  if (out_len < needed) {
    return -2;
  }
  if (needed != 0 && out == nullptr) {
    return -1;
  }

  char* cursor = out;
  for (const std::string& object_key : object_keys) {
    std::memcpy(cursor, object_key.data(), object_key.size());
    cursor += object_key.size();
    *cursor = '\0';
    ++cursor;
  }
  return 1;
}
}
