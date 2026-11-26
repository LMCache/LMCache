// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
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

// SHA-256 for the long-name fallback. The format is shared with
// fs_l2_adapter.py::_object_key_to_filename, so both sides must produce the
// same digest; a real crypto library would be a build dependency this header
// does not otherwise need.
namespace detail {

inline uint32_t rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}

inline std::string sha256_hex(const std::string& input) {
  static const uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
      0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
      0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
      0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
      0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
      0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
      0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
      0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::vector<uint8_t> msg(input.begin(), input.end());
  const uint64_t bit_len = static_cast<uint64_t>(msg.size()) * 8;
  msg.push_back(0x80);
  while (msg.size() % 64 != 56) msg.push_back(0);
  for (int i = 7; i >= 0; --i) msg.push_back((bit_len >> (i * 8)) & 0xff);
  for (size_t off = 0; off < msg.size(); off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (uint32_t(msg[off + i * 4]) << 24) |
             (uint32_t(msg[off + i * 4 + 1]) << 16) |
             (uint32_t(msg[off + i * 4 + 2]) << 8) |
             uint32_t(msg[off + i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t t1 = hh + s1 + ch + k[i] + w[i];
      uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t t2 = s0 + maj;
      hh = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += hh;
  }
  static const char* hex = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (int i = 0; i < 8; ++i) {
    for (int shift = 28; shift >= 0; shift -= 4) {
      out.push_back(hex[(h[i] >> shift) & 0xf]);
    }
  }
  return out;
}

}  // namespace detail

std::string key_to_filename(const std::string& key,
                            bool shard_directories = false) {
  // Input key format (from _object_key_to_string):
  // Unsalted:
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>
  // Salted:
  //     <model_name>@<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>
  //
  // Extended keys append a sixth field: @<cache_salt>@x1-<identity_token>.
  // The salt placeholder is present even when empty.
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

  // A sixth field carries the versioned, path-safe extension identity token.
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 4 && parts.size() != 5 && parts.size() != 6) {
    throw std::runtime_error(
        "FSConnector: malformed key (expected 4, 5 or 6 '@'-separated "
        "fields): " +
        key);
  }

  const std::string& model_name = parts[0];
  const std::string& kv_rank_hex = parts[1];
  const std::string& object_group_id_hex = parts[2];
  const std::string& chunk_hash = parts[3];
  const std::string cache_salt = parts.size() >= 5 ? parts[4] : std::string();

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
  if (!cache_salt.empty() || parts.size() == 6) {
    result += KEY_SEP;
    result += cache_salt;
  }
  if (parts.size() == 6) {
    result += KEY_SEP;
    result += parts[5];
  }
  result += FILE_EXT;

  // A name that no longer fits NAME_MAX cannot be written, and the key
  // encoding is reversible only while it fits. Both languages fall back to
  // the same digest name so a cache directory stays readable across the
  // native and Python adapters.
  static constexpr size_t kNameMaxBytes = 255;
  if (result.size() > kNameMaxBytes) {
    result = "l-" + detail::sha256_hex(result) + FILE_EXT;
  }

  if (!shard_directories) return result;
  if (chunk_hash.size() < 4) {
    throw std::runtime_error(
        "shard_dirs requires a chunk hash of at least two bytes");
  }
  return chunk_hash.substr(0, 2) + "/" + chunk_hash.substr(2, 2) + "/" + result;
}

// Inverse of key_to_filename(): convert a flat (non-sharded) filename back to
// wire key.
//
// The flat (non-sharded) filename shape produced by key_to_filename():
//   <safe_model>@0x<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>[@x1-<identity_token>]].data
// Returns the wire key:
//   <model>@<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>[@x1-<identity_token>]]
//
// Returns an empty string for any file that is not a well-formed KV-key
// data file (e.g.: wrong extension, fewer than 4 '@'-separated fields).
std::string filename_to_key(const std::string& filename) {
  const std::string ext(FILE_EXT);
  if (filename.size() <= ext.size() ||
      filename.compare(filename.size() - ext.size(), ext.size(), ext) != 0) {
    return {};
  }

  // Strip the ".data" suffix
  const std::string body = filename.substr(0, filename.size() - ext.size());

  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= body.size(); ++pos) {
    if (pos == body.size() || body[pos] == KEY_SEP) {
      parts.emplace_back(body.substr(start, pos - start));
      start = pos + 1;
    }
  }
  // Need model, kv_rank, ogid and chunk_hash (4 fields minimum).
  if (parts.size() != 4 && parts.size() != 5 && parts.size() != 6) {
    return {};
  }

  // Restore '/' in model_name
  std::string model = parts[0];
  const std::string sep(PATH_SLASH_REPLACEMENT);
  size_t spos = 0;
  while ((spos = model.find(sep, spos)) != std::string::npos) {
    model.replace(spos, sep.size(), "/");
    spos += 1;
  }

  // Drop the "0x" prefix from kv_rank.
  std::string kv_rank = parts[1];
  if (kv_rank.size() >= 2 && kv_rank.substr(0, 2) == "0x") {
    kv_rank = kv_rank.substr(2);
  }

  // Reassemble the wire key.
  std::string result = model;
  result += KEY_SEP;
  result += kv_rank;
  for (size_t i = 2; i < parts.size(); ++i) {
    result += KEY_SEP;
    result += parts[i];
  }
  return result;
}

// Extract the ``chunk_hash`` field from key.
//
// Wire key format:
//   <model>@<kv_rank_hex>@<ogid_hex>@<chunk_hash_hex>[@<cache_salt>[@x1-<identity_token>]]
//
// The chunk_hash is always the 4th '@'-separated field (index 3).
std::string get_chunk_hash_from_key(const std::string& key) {
  std::vector<std::string> parts;
  size_t start = 0;
  for (size_t pos = 0; pos <= key.size(); ++pos) {
    if (pos == key.size() || key[pos] == KEY_SEP) {
      parts.emplace_back(key.substr(start, pos - start));
      start = pos + 1;
    }
  }
  if (parts.size() != 4 && parts.size() != 5 && parts.size() != 6) {
    throw std::runtime_error(
        "Malformed key (expected 4, 5 or 6 '@'-separated fields): " + key);
  }
  if (parts[3].empty()) {
    throw std::runtime_error(
        "Malformed key (chunk hash field which behind the third '@' is "
        "empty): " +
        key);
  }
  return parts[3];
}
}  // namespace connector
}  // namespace lmcache
