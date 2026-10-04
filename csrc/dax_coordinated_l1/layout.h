// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared on-media metadata and payload layout ABI.

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lmcache::dax_coordinated_l1 {

using Digest = std::array<std::uint8_t, 32>;

constexpr std::size_t kCacheLineBytes = 64;
constexpr std::size_t kSuperblockBytes = 384;
constexpr std::size_t kMaxLevels = 8;
constexpr std::uint32_t kFormatVersion = 2;
constexpr std::uint32_t kVisibilityX86Clflush64BV1 = 1;
constexpr std::uint32_t kVisibilityX86ClflushoptBulkV1 = 2;
constexpr std::uint32_t kPayloadMappingSeparate = 1;
constexpr std::array<char, 8> kMagic = {'L', 'M', 'C', 'D',
                                        'A', 'X', '1', '\0'};

enum class BucketState : std::uint8_t {
  kFree = 0,
  kReady = 1,
  kWriteLocked = 2,
  kRecovery = 3,
};

enum class WriteIntent : std::uint8_t {
  kNone = 0,
  kAllocating = 1,
  // Value 2 is reserved for a retired write intent.
  kInvalidating = 3,
};

enum class ReaderGate : std::uint8_t {
  kClosed = 0,
  kOpen = 1,
};

enum class PayloadAllocationState : std::uint8_t {
  kFree = 0,
  kAllocated = 1,
};

struct alignas(64) Superblock {
  std::array<char, 8> magic;
  std::uint32_t format_version;
  std::uint32_t header_bytes;
  std::uint64_t region_epoch;
  std::uint32_t max_participants;
  std::uint32_t level_count;
  std::array<std::uint64_t, kMaxLevels> buckets_per_level;
  std::uint32_t cache_line_bytes;
  std::uint32_t visibility_mode;
  std::uint32_t payload_mapping_mode;
  std::uint32_t
      reserved0;  // Explicit padding keeps checksum bytes deterministic.
  std::uint64_t payload_slot_bytes;
  std::uint64_t payload_slot_count;
  std::uint64_t payload_alignment;
  std::uint64_t region_size_bytes;
  std::uint64_t payload_region_size_bytes;
  std::uint64_t participant_registry_offset;
  std::array<std::uint64_t, kMaxLevels> bucket_metadata_offsets;
  std::uint64_t bucket_peterson_offset;
  std::uint64_t reader_activity_offset;
  std::uint64_t payload_header_offset;
  std::uint64_t payload_data_offset;
  Digest layout_profile_digest;
  Digest hardware_qualification_digest;
  Digest region_id_digest;
  std::uint64_t checksum;
  std::array<std::uint8_t, 24> reserved;
};

struct alignas(64) ParticipantEntry {
  std::uint32_t participant_id;
  std::uint32_t participant_state;
  std::array<std::uint8_t, 8> reserved0;
  std::uint64_t owner_slot_begin;
  std::uint64_t owner_slot_count;
  Digest reserved1;
};

struct alignas(64) BucketIdentityLine {
  Digest key_digest;
  std::uint64_t bucket_generation;
  std::uint64_t global_payload_slot_id;
  std::uint64_t slot_generation;
  std::uint32_t payload_length;
  std::uint32_t layout_id;
};

struct alignas(64) BucketControlLine {
  // Keep legacy counter/writer bytes reserved; neither participates in
  // admission.
  std::array<std::uint8_t, 20> reserved0;
  BucketState state;
  WriteIntent write_intent;
  ReaderGate reader_gate;
  std::array<std::uint8_t, 41> reserved;
};

struct alignas(64) BucketMetaSlot {
  BucketIdentityLine identity;
  BucketControlLine control;
};

struct alignas(64) CacheLineWord {
  std::uint64_t value;
  std::array<std::uint8_t, 56> reserved;
};

struct alignas(64) BucketPetersonRecord {
  std::array<CacheLineWord, 2> participant_flag;
  CacheLineWord victim;
};

struct alignas(64) PackedBitWord {
  std::array<std::uint64_t, 8> words;
};

struct alignas(64) PayloadSlotHeader {
  std::uint32_t owner_participant_id;
  std::array<std::uint8_t, 12> reserved0;
  std::uint64_t slot_generation;
  PayloadAllocationState allocation_state;
  std::array<std::uint8_t, 7> reserved1;
  std::uint64_t back_bucket_id;
  std::uint64_t back_bucket_generation;
  std::uint32_t payload_length;
  std::array<std::uint8_t, 12> reserved2;
};

struct LayoutDescription {
  std::uint64_t participant_registry_offset;
  std::array<std::uint64_t, kMaxLevels> bucket_metadata_offsets{};
  std::uint64_t bucket_peterson_offset;
  std::uint64_t reader_activity_offset;
  std::uint64_t payload_header_offset;
  std::uint64_t total_bucket_count;
  std::uint64_t bit_word_count_per_participant;
  std::uint64_t required_metadata_bytes;
  std::uint64_t required_payload_bytes;
};

inline LayoutDescription calculate_layout(
    const std::vector<std::uint64_t>& buckets_per_level,
    std::uint64_t payload_slot_count, std::uint64_t payload_slot_bytes,
    std::uint64_t payload_alignment, std::uint32_t participant_count = 2) {
  if (participant_count != 2 && participant_count != 4) {
    throw std::invalid_argument("participant count must be 2 or 4");
  }
  if (buckets_per_level.empty() || buckets_per_level.size() > kMaxLevels) {
    throw std::invalid_argument("level count must be in [1, 8]");
  }
  if (payload_slot_count == 0 || payload_slot_bytes == 0) {
    throw std::invalid_argument("payload slot count and size must be positive");
  }
  if (payload_alignment < kCacheLineBytes ||
      (payload_alignment & (payload_alignment - 1)) != 0) {
    throw std::invalid_argument(
        "payload alignment must be a power of two and at least 64");
  }

  LayoutDescription result{};
  std::uint64_t offset = kSuperblockBytes;
  auto section = [&](std::uint64_t count, std::uint64_t width) {
    if (count > (std::numeric_limits<std::uint64_t>::max() - offset) / width) {
      throw std::overflow_error("Device-DAX metadata size overflow");
    }
    const auto begin = offset;
    offset += count * width;
    return begin;
  };
  result.participant_registry_offset =
      section(participant_count, sizeof(ParticipantEntry));
  for (std::size_t level = 0; level < buckets_per_level.size(); ++level) {
    if (buckets_per_level[level] == 0) {
      throw std::invalid_argument("bucket counts must be positive");
    }
    result.bucket_metadata_offsets[level] =
        section(buckets_per_level[level], sizeof(BucketMetaSlot));
    result.total_bucket_count += buckets_per_level[level];
  }
  result.bucket_peterson_offset =
      section(result.total_bucket_count,
              (participant_count - 1) * sizeof(BucketPetersonRecord));
  result.bit_word_count_per_participant = (payload_slot_count - 1) / 512 + 1;
  result.reader_activity_offset =
      section(result.bit_word_count_per_participant,
              participant_count * sizeof(PackedBitWord));
  result.payload_header_offset =
      section(payload_slot_count, sizeof(PayloadSlotHeader));
  result.required_metadata_bytes = offset;
  if (payload_slot_count >
      std::numeric_limits<std::uint64_t>::max() / payload_slot_bytes) {
    throw std::overflow_error("Device-DAX payload size overflow");
  }
  result.required_payload_bytes = payload_slot_count * payload_slot_bytes;
  return result;
}

static_assert(sizeof(BucketState) == 1);
static_assert(sizeof(WriteIntent) == 1);
static_assert(sizeof(ReaderGate) == 1);
static_assert(sizeof(PayloadAllocationState) == 1);
static_assert(offsetof(BucketControlLine, state) == 20);
static_assert(offsetof(PayloadSlotHeader, allocation_state) == 24);

static_assert(sizeof(Superblock) == kSuperblockBytes);
static_assert(alignof(Superblock) == kCacheLineBytes);
static_assert(sizeof(ParticipantEntry) == kCacheLineBytes);
static_assert(sizeof(BucketIdentityLine) == kCacheLineBytes);
static_assert(sizeof(BucketControlLine) == kCacheLineBytes);
static_assert(sizeof(BucketMetaSlot) == 2 * kCacheLineBytes);
static_assert(sizeof(CacheLineWord) == kCacheLineBytes);
static_assert(sizeof(BucketPetersonRecord) == 3 * kCacheLineBytes);
static_assert(sizeof(PackedBitWord) == kCacheLineBytes);
static_assert(sizeof(PayloadSlotHeader) == kCacheLineBytes);

}  // namespace lmcache::dax_coordinated_l1
