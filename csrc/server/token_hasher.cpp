// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — blake3 token hashing implementation

#include "token_hasher.h"

#include <blake3.h>

#include <cassert>
#include <cstring>

namespace lmcache {
namespace server {

// ----------------------------------------------------------------------------
// Byte-order helpers (big-endian serialization)
// ----------------------------------------------------------------------------

namespace {

/// Write a signed 64-bit int as 8 bytes big-endian (matches Python's
/// int.to_bytes(8, 'big', signed=True)).
inline void write_be_i64(uint8_t* dst, int64_t val) {
  auto uval = static_cast<uint64_t>(val);
  dst[0] = static_cast<uint8_t>(uval >> 56);
  dst[1] = static_cast<uint8_t>(uval >> 48);
  dst[2] = static_cast<uint8_t>(uval >> 40);
  dst[3] = static_cast<uint8_t>(uval >> 32);
  dst[4] = static_cast<uint8_t>(uval >> 24);
  dst[5] = static_cast<uint8_t>(uval >> 16);
  dst[6] = static_cast<uint8_t>(uval >> 8);
  dst[7] = static_cast<uint8_t>(uval);
}

/// Write a 32-bit unsigned int as 4 bytes big-endian (matches Python's
/// struct.pack('>I', val)).
inline void write_be_u32(uint8_t* dst, uint32_t val) {
  dst[0] = static_cast<uint8_t>(val >> 24);
  dst[1] = static_cast<uint8_t>(val >> 16);
  dst[2] = static_cast<uint8_t>(val >> 8);
  dst[3] = static_cast<uint8_t>(val);
}

}  // namespace

// ----------------------------------------------------------------------------
// TokenHasher
// ----------------------------------------------------------------------------

TokenHasher::TokenHasher(int chunk_size) : chunk_size_(chunk_size) {
  init_none_hash();
}

TokenHasher::~TokenHasher() = default;

void TokenHasher::init_none_hash() {
  // Python equivalent:
  //   hash_func((0, (0,), None))
  //   prefix_hash = (0).to_bytes(8, 'big', signed=True)  → 8 zero bytes
  //   tokens = struct.pack('>1I', 0)                      → 4 zero bytes

  blake3_hasher hasher;
  blake3_hasher_init(&hasher);

  // prefix_hash = 0 as 8 bytes big-endian signed
  uint8_t prefix_buf[8];
  write_be_i64(prefix_buf, 0);
  blake3_hasher_update(&hasher, prefix_buf, 8);

  // tokens = (0,) as one big-endian uint32
  uint8_t token_buf[4];
  write_be_u32(token_buf, 0);
  blake3_hasher_update(&hasher, token_buf, 4);

  none_hash_.resize(BLAKE3_OUT_LEN);
  blake3_hasher_finalize(&hasher, none_hash_.data(), BLAKE3_OUT_LEN);
}

HashBytes TokenHasher::hash_tokens(const int32_t* tokens, size_t count,
                                   const HashBytes& prefix_hash) const {
  blake3_hasher hasher;
  blake3_hasher_init(&hasher);

  // Serialize prefix_hash.
  // In Python, when prefix_hash is bytes, it does h.update(prefix_hash).
  // When it's an int, it serializes as 8 bytes big-endian.
  // In C++ our HashBytes is always the raw 32-byte digest, so we feed it
  // directly (matches the Python `bytes(prefix_hash)` / `h.update(prefix_hash)`
  // path).
  const HashBytes& pfx = prefix_hash.empty() ? none_hash_ : prefix_hash;
  blake3_hasher_update(&hasher, pfx.data(), pfx.size());

  // Serialize token IDs as big-endian uint32 array.
  // Python: struct.pack(f'>{len(tokens)}I', *tokens)
  // We allocate a temporary buffer for the big-endian encoding.
  std::vector<uint8_t> token_buf(count * 4);
  for (size_t i = 0; i < count; ++i) {
    write_be_u32(token_buf.data() + i * 4, static_cast<uint32_t>(tokens[i]));
  }
  blake3_hasher_update(&hasher, token_buf.data(), token_buf.size());

  HashBytes result(BLAKE3_OUT_LEN);
  blake3_hasher_finalize(&hasher, result.data(), BLAKE3_OUT_LEN);
  return result;
}

std::vector<HashBytes> TokenHasher::compute_chunk_hashes(
    const std::vector<int32_t>& token_ids, int start, int end) const {
  std::vector<HashBytes> hashes;

  int effective_len = (end >= 0)
                          ? std::min(static_cast<int>(token_ids.size()), end)
                          : static_cast<int>(token_ids.size());

  // Truncate to complete chunks
  int num_complete = effective_len - (effective_len % chunk_size_);

  HashBytes prefix = none_hash_;
  for (int i = 0; i < num_complete; i += chunk_size_) {
    prefix = hash_tokens(token_ids.data() + i, static_cast<size_t>(chunk_size_),
                         prefix);
    if (i >= start) {
      hashes.push_back(prefix);
    }
  }
  return hashes;
}

}  // namespace server
}  // namespace lmcache
