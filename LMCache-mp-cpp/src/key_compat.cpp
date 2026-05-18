// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/key_compat.h"

#if __has_include(<llvm-c-18/llvm-c/blake3.h>)
  #include <llvm-c-18/llvm-c/blake3.h>
#else
  #include <llvm-c/blake3.h>
#endif

#include <algorithm>
#include <array>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {

using HashBytes = std::array<std::uint8_t, LMCACHE_MP_CPP_BLAKE3_OUT_LEN>;

void append_u32_be(std::vector<std::uint8_t>& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>((value >> 24) & 0xff));
  out.push_back(static_cast<std::uint8_t>((value >> 16) & 0xff));
  out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
  out.push_back(static_cast<std::uint8_t>(value & 0xff));
}

void hash_bytes(const std::uint8_t* prefix, std::uint64_t prefix_len,
                const std::uint32_t* tokens, std::uint64_t token_count,
                std::uint8_t* out32) {
  llvm_blake3_hasher hasher;
  llvm_blake3_hasher_init(&hasher);
  llvm_blake3_hasher_update(&hasher, prefix,
                            static_cast<std::size_t>(prefix_len));

  std::vector<std::uint8_t> packed;
  packed.reserve(static_cast<std::size_t>(token_count * sizeof(std::uint32_t)));
  for (std::uint64_t i = 0; i < token_count; ++i) {
    append_u32_be(packed, tokens[i]);
  }
  if (!packed.empty()) {
    llvm_blake3_hasher_update(&hasher, packed.data(), packed.size());
  }
  llvm_blake3_hasher_finalize(&hasher, out32, LMCACHE_MP_CPP_BLAKE3_OUT_LEN);
}

HashBytes none_hash() {
  static constexpr std::uint8_t kZeroPrefix[8] = {};
  static constexpr std::uint32_t kZeroToken[1] = {0};
  HashBytes out{};
  hash_bytes(kZeroPrefix, sizeof(kZeroPrefix), kZeroToken, 1, out.data());
  return out;
}

std::string hex_encode(const std::uint8_t* bytes, std::uint64_t len) {
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (std::uint64_t i = 0; i < len; ++i) {
    out << std::setw(2) << static_cast<unsigned>(bytes[i]);
  }
  return out.str();
}

}  // namespace

extern "C" {

int lmcache_mp_cpp_blake3_none_hash(std::uint8_t* out32) {
  if (out32 == nullptr) {
    return -1;
  }
  const HashBytes out = none_hash();
  std::memcpy(out32, out.data(), out.size());
  return 1;
}

int lmcache_mp_cpp_blake3_hash_tokens(const std::uint8_t* prefix,
                                      std::uint64_t prefix_len,
                                      const std::uint32_t* tokens,
                                      std::uint64_t token_count,
                                      std::uint8_t* out32) {
  if (prefix == nullptr || tokens == nullptr || out32 == nullptr) {
    return -1;
  }
  if (prefix_len != 8 && prefix_len != LMCACHE_MP_CPP_BLAKE3_OUT_LEN) {
    return -2;
  }
  hash_bytes(prefix, prefix_len, tokens, token_count, out32);
  return 1;
}

int lmcache_mp_cpp_blake3_chunk_hashes(
    const std::uint32_t* tokens, std::uint64_t token_count,
    std::uint64_t chunk_size, std::uint64_t start, std::uint64_t end,
    std::uint8_t* out, std::uint64_t max_hashes, std::uint64_t* out_hashes) {
  if (tokens == nullptr || out == nullptr || out_hashes == nullptr ||
      chunk_size == 0 || start % chunk_size != 0 || end % chunk_size != 0 ||
      start > end) {
    return -1;
  }

  const std::uint64_t effective_len = std::min(token_count, end);
  const std::uint64_t num_complete =
      effective_len - (effective_len % chunk_size);
  HashBytes prefix = none_hash();
  std::uint64_t written = 0;
  for (std::uint64_t i = 0; i < num_complete; i += chunk_size) {
    HashBytes next{};
    hash_bytes(prefix.data(), prefix.size(), tokens + i, chunk_size,
               next.data());
    prefix = next;
    if (i >= start) {
      if (written >= max_hashes) {
        return -2;
      }
      std::memcpy(out + written * LMCACHE_MP_CPP_BLAKE3_OUT_LEN, prefix.data(),
                  prefix.size());
      ++written;
    }
  }
  *out_hashes = written;
  return 1;
}

std::uint32_t lmcache_mp_cpp_compute_kv_rank(std::uint32_t world_size,
                                             std::uint32_t global_rank,
                                             std::uint32_t local_world_size,
                                             std::uint32_t local_rank) {
  return (world_size << 24) | (global_rank << 16) | (local_world_size << 8) |
         local_rank;
}

int lmcache_mp_cpp_expand_kv_ranks(std::uint32_t world_size,
                                   std::int32_t worker_id,
                                   std::uint32_t* out_ranks,
                                   std::uint64_t max_ranks,
                                   std::uint64_t* out_count) {
  if (out_ranks == nullptr || out_count == nullptr || world_size == 0) {
    return -1;
  }

  if (worker_id < 0) {
    if (max_ranks < world_size) {
      return -2;
    }
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      out_ranks[rank] =
          lmcache_mp_cpp_compute_kv_rank(world_size, rank, world_size, rank);
    }
    *out_count = world_size;
    return 1;
  }

  const auto rank = static_cast<std::uint32_t>(worker_id);
  if (rank >= world_size || max_ranks < 1) {
    return -2;
  }
  out_ranks[0] =
      lmcache_mp_cpp_compute_kv_rank(world_size, rank, world_size, rank);
  *out_count = 1;
  return 1;
}

int lmcache_mp_cpp_object_key_string(const char* model_name,
                                     std::uint32_t kv_rank,
                                     const std::uint8_t* chunk_hash,
                                     std::uint64_t chunk_hash_len,
                                     const char* cache_salt, char* out,
                                     std::uint64_t out_len,
                                     std::uint64_t* needed_len) {
  if (model_name == nullptr || chunk_hash == nullptr || out == nullptr ||
      needed_len == nullptr) {
    return -1;
  }
  std::ostringstream encoded;
  encoded << model_name << "@" << std::hex << std::setw(8) << std::setfill('0')
          << kv_rank << "@" << hex_encode(chunk_hash, chunk_hash_len);
  if (cache_salt != nullptr && std::strlen(cache_salt) > 0) {
    encoded << "@" << cache_salt;
  }
  const std::string value = encoded.str();
  *needed_len = value.size();
  if (out_len <= value.size()) {
    return -2;
  }
  std::memcpy(out, value.data(), value.size());
  out[value.size()] = '\0';
  return 1;
}
}
