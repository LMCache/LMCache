// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace lmcache {
namespace connector {

static constexpr char KEY_SEP = '@';
static constexpr const char* PATH_SLASH_REPLACEMENT = "-SEP-";
static constexpr const char* FILE_EXT = ".data";

std::string replace_all(const std::string& str, const std::string& from,
                        const std::string& to) {
  std::string result = str;
  size_t pos = 0;
  while ((pos = result.find(from, pos)) != std::string::npos) {
    result.replace(pos, from.size(), to);
    pos += to.size();
  }
  return result;
}

std::string key_to_filename(const std::string& key,
                            bool shard_directories = false) {
  // Input key format (from _object_key_to_string):
  //   Unsalted: <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>
  //   Salted  : <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>
  //
  // Output filename (matching fs_l2_adapter.py._object_key_to_filename):
  // Unsalted::
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>.data
  // Salted (trailing ``cache_salt``)::
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>.data
  //
  // The unsalted 4-field shape is bit-identical to the pre-cache_salt
  // format, so existing cache directories remain valid.
  //
  // NOTE: both model_name and cache_salt are forbidden from containing
  // '@' (invariant enforced on the Python side), so splitting on '@'
  // is unambiguous — no marker, no rsplit.

  // Split on '@' — must yield 4 (unsalted) or 5 (salted) fields.
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 4 && parts.size() != 5) {
    throw std::runtime_error(
        "FSConnector: malformed key (expected 4 or 5 '@'-separated fields): " +
        key);
  }

  const std::string& model_name = parts[0];
  const std::string& kv_rank_hex = parts[1];
  const std::string& object_group_id_hex = parts[2];
  const std::string& chunk_hash = parts[3];
  const std::string cache_salt = parts.size() == 5 ? parts[4] : std::string();

  // Replace '/' with '-SEP-' for filesystem safety
  std::string safe_model = replace_all(model_name, "/", PATH_SLASH_REPLACEMENT);

  // Emit filename. Salt is appended at the tail so the unsalted shape
  // matches what older builds wrote to disk.
  std::string result;
  result.reserve(safe_model.size() + kv_rank_hex.size() +
                 object_group_id_hex.size() + chunk_hash.size() +
                 cache_salt.size() + 32);
  result += safe_model;
  result += KEY_SEP;
  result += "0x";
  result += kv_rank_hex;
  result += KEY_SEP;
  result += object_group_id_hex;
  result += KEY_SEP;
  result += chunk_hash;
  if (!cache_salt.empty()) {
    result += KEY_SEP;
    result += cache_salt;
  }
  result += FILE_EXT;

  if (!shard_directories) return result;
  if (chunk_hash.size() < 4) {
    throw std::runtime_error(
        "shard_dirs requires a chunk hash of at least two bytes");
  }
  return chunk_hash.substr(0, 2) + "/" + chunk_hash.substr(2, 2) + "/" + result;
}
}  // namespace connector
}  // namespace lmcache