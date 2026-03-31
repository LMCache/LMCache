// SPDX-License-Identifier: Apache-2.0
// Pure C++ LMCache Server — blake3 token hashing
//
// Rolling prefix hash computation for token chunks.
// Each chunk's hash depends on all previous chunks (prefix chain).

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace lmcache {
namespace server {

// 32-byte blake3 hash value
using HashBytes = std::vector<uint8_t>;

// ============================================================================
// TokenHasher — rolling prefix hash for token chunks
// ============================================================================

class TokenHasher {
 public:
  /// @param chunk_size  Number of tokens per chunk (e.g. 256)
  explicit TokenHasher(int chunk_size = 256);
  ~TokenHasher();

  // Non-copyable
  TokenHasher(const TokenHasher&) = delete;
  TokenHasher& operator=(const TokenHasher&) = delete;

  /// Compute rolling prefix hashes for all complete chunks.
  ///
  /// The rolling hash is always computed from the beginning of token_ids
  /// (since each chunk's hash depends on all previous chunks), but only
  /// hashes for chunks within [start, end) are returned.
  ///
  /// @param token_ids   Full token sequence
  /// @param start       Token-level start index (chunk-aligned, default 0)
  /// @param end         Token-level end index (chunk-aligned, -1 = all)
  /// @return            Hash bytes for each chunk in [start, end)
  std::vector<HashBytes> compute_chunk_hashes(
      const std::vector<int32_t>& token_ids, int start = 0, int end = -1) const;

  /// Hash a single chunk with the given prefix hash context.
  /// @param tokens       Token IDs for this chunk
  /// @param prefix_hash  Hash of all preceding chunks (empty = none_hash)
  /// @return             32-byte blake3 hash
  HashBytes hash_tokens(const int32_t* tokens, size_t count,
                        const HashBytes& prefix_hash) const;

  /// The initial prefix hash (hash of the "none" sentinel).
  /// Equivalent to Python's init_none_hash().
  const HashBytes& none_hash() const { return none_hash_; }

  int chunk_size() const { return chunk_size_; }

 private:
  int chunk_size_;
  HashBytes none_hash_;

  /// Compute the none_hash value
  void init_none_hash();
};

}  // namespace server
}  // namespace lmcache
