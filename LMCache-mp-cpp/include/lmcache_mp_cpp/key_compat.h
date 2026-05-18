// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

static constexpr std::size_t LMCACHE_MP_CPP_BLAKE3_OUT_LEN = 32;

int lmcache_mp_cpp_blake3_none_hash(std::uint8_t* out32);

int lmcache_mp_cpp_blake3_hash_tokens(const std::uint8_t* prefix,
                                      std::uint64_t prefix_len,
                                      const std::uint32_t* tokens,
                                      std::uint64_t token_count,
                                      std::uint8_t* out32);

int lmcache_mp_cpp_blake3_chunk_hashes(
    const std::uint32_t* tokens, std::uint64_t token_count,
    std::uint64_t chunk_size, std::uint64_t start, std::uint64_t end,
    std::uint8_t* out, std::uint64_t max_hashes, std::uint64_t* out_hashes);

std::uint32_t lmcache_mp_cpp_compute_kv_rank(std::uint32_t world_size,
                                             std::uint32_t global_rank,
                                             std::uint32_t local_world_size,
                                             std::uint32_t local_rank);

int lmcache_mp_cpp_expand_kv_ranks(std::uint32_t world_size,
                                   std::int32_t worker_id,
                                   std::uint32_t* out_ranks,
                                   std::uint64_t max_ranks,
                                   std::uint64_t* out_count);

int lmcache_mp_cpp_object_key_string(const char* model_name,
                                     std::uint32_t kv_rank,
                                     const std::uint8_t* chunk_hash,
                                     std::uint64_t chunk_hash_len,
                                     const char* cache_salt, char* out,
                                     std::uint64_t out_len,
                                     std::uint64_t* needed_len);
}
