// SPDX-License-Identifier: Apache-2.0

#include "dax_coordinated_l1/index_core.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <unordered_set>

#include "dax_coordinated_l1/peterson.h"
#include "dax_coordinated_l1/visibility_x86.h"

namespace lmcache::dax_coordinated_l1 {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

std::uint64_t fnv1a(const void* data, std::size_t length,
                    std::uint64_t seed = kFnvOffset) {
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  auto value = seed;
  for (std::size_t index = 0; index < length; ++index) {
    value ^= bytes[index];
    value *= kFnvPrime;
  }
  return value;
}

std::uint64_t superblock_checksum(const Superblock& input) {
  Superblock copy = input;
  copy.checksum = 0;
  return fnv1a(&copy, sizeof(copy));
}

bool valid_state(BucketState state, WriteIntent intent, ReaderGate gate) {
  if (state == BucketState::kFree) {
    return intent == WriteIntent::kNone && gate == ReaderGate::kClosed;
  }
  if (state == BucketState::kReady) {
    return intent == WriteIntent::kNone && gate == ReaderGate::kOpen;
  }
  if (state == BucketState::kWriteLocked) {
    return (intent == WriteIntent::kAllocating ||
            intent == WriteIntent::kInvalidating) &&
           gate == ReaderGate::kClosed;
  }
  if (state == BucketState::kRecovery) {
    return gate == ReaderGate::kClosed;
  }
  return false;
}

void publish_reader_activity(
    const std::unordered_set<PackedBitWord*>& dirty_words) {
  // One cache line covers 512 payload slots. Batch publication keeps one
  // flush per changed line even when payload addresses are widely separated.
  for (auto* word : dirty_words) {
    publish_range(word, sizeof(PackedBitWord));
  }
}

Superblock make_superblock(const FormatParameters& parameters,
                           std::uint64_t region_size,
                           const LayoutDescription& layout) {
  parameters.owner_range(0);  // Validate ownership before touching the mapping.
  if (parameters.hardware_qualification_digest == Digest{}) {
    throw std::invalid_argument("hardware qualification digest is required");
  }
  if ((parameters.layout_profile_digest == Digest{}) ||
      (parameters.region_id_digest == Digest{})) {
    throw std::invalid_argument("layout and region digests are required");
  }
  if (parameters.visibility_mode != kVisibilityX86Clflush64BV1 &&
      parameters.visibility_mode != kVisibilityX86ClflushoptBulkV1) {
    throw std::invalid_argument("unsupported payload visibility mode");
  }
  if (parameters.visibility_mode == kVisibilityX86ClflushoptBulkV1 &&
      !query_cpu_visibility_profile().has_clflushopt) {
    throw std::invalid_argument(
        "bulk payload visibility requires CPUID CLFLUSHOPT support");
  }
  if (layout.required_metadata_bytes > region_size) {
    throw std::invalid_argument(
        "configured metadata exceeds the mapped region");
  }
  if (layout.required_payload_bytes > parameters.payload_region_size) {
    throw std::invalid_argument("configured slots exceed the payload region");
  }
  Superblock header;
  std::memset(&header, 0, sizeof(header));
  header.magic = kMagic;
  header.format_version = kFormatVersion;
  header.header_bytes = kSuperblockBytes;
  header.region_epoch = parameters.region_epoch;
  header.max_participants = parameters.participant_count;
  header.level_count = parameters.buckets_per_level.size();
  for (std::size_t level = 0; level < parameters.buckets_per_level.size();
       ++level) {
    header.buckets_per_level[level] = parameters.buckets_per_level[level];
    header.bucket_metadata_offsets[level] =
        layout.bucket_metadata_offsets[level];
  }
  header.cache_line_bytes = kCacheLineBytes;
  header.visibility_mode = parameters.visibility_mode;
  header.payload_mapping_mode = kPayloadMappingSeparate;
  header.payload_slot_bytes = parameters.payload_slot_bytes;
  header.payload_slot_count = parameters.payload_slot_count;
  header.payload_alignment = parameters.payload_alignment;
  header.region_size_bytes = region_size;
  header.payload_region_size_bytes = parameters.payload_region_size;
  header.participant_registry_offset = layout.participant_registry_offset;
  header.bucket_peterson_offset = layout.bucket_peterson_offset;
  header.reader_activity_offset = layout.reader_activity_offset;
  header.payload_header_offset = layout.payload_header_offset;
  header.payload_data_offset = 0;
  header.layout_profile_digest = parameters.layout_profile_digest;
  header.hardware_qualification_digest =
      parameters.hardware_qualification_digest;
  header.region_id_digest = parameters.region_id_digest;

  header.checksum = superblock_checksum(header);
  return header;
}

}  // namespace

void format_region(void* base, std::uint64_t region_size,
                   const FormatParameters& parameters) {
  validate_cpu_visibility_profile();
  if (base == nullptr ||
      reinterpret_cast<std::uintptr_t>(base) % kCacheLineBytes != 0) {
    throw std::invalid_argument("mapped base must be 64-byte aligned");
  }
  const auto layout = parameters.layout();
  auto header = make_superblock(parameters, region_size, layout);
  header.magic = {};  // Publish completion only after every metadata write.
  auto* bytes = static_cast<std::uint8_t*>(base);
  std::memset(bytes, 0, layout.required_metadata_bytes);
  auto* superblock = reinterpret_cast<Superblock*>(bytes);
  *superblock = header;

  auto* participants = reinterpret_cast<ParticipantEntry*>(
      bytes + layout.participant_registry_offset);
  for (std::size_t participant = 0; participant < parameters.participant_count;
       ++participant) {
    participants[participant].participant_id = participant;
    participants[participant].participant_state = 1;
    const auto [begin, count] = parameters.owner_range(participant);
    participants[participant].owner_slot_begin = begin;
    participants[participant].owner_slot_count = count;
  }

  auto* headers = reinterpret_cast<PayloadSlotHeader*>(
      bytes + layout.payload_header_offset);
  for (std::uint64_t slot = 0; slot < parameters.payload_slot_count; ++slot) {
    for (std::uint32_t owner = 0; owner < parameters.participant_count;
         ++owner) {
      const auto [begin, count] = parameters.owner_range(owner);
      if (slot >= begin && slot - begin < count) {
        headers[slot].owner_participant_id = owner;
        break;
      }
    }
    headers[slot].slot_generation = 1;
    headers[slot].allocation_state = PayloadAllocationState::kFree;
    headers[slot].back_bucket_id = std::numeric_limits<std::uint64_t>::max();
  }
  if (parameters.visibility_mode == kVisibilityX86ClflushoptBulkV1) {
    publish_range_bulk(bytes, layout.required_metadata_bytes);
  } else {
    publish_range(bytes, layout.required_metadata_bytes);
  }
  // The preceding publication fences every metadata write. The final image
  // (including its checksum) is unchanged; only its publication is ordered.
  superblock->magic = kMagic;
  publish_range(superblock, kCacheLineBytes);
}

std::uint64_t bucket_id_for_level(const Superblock& superblock,
                                  const Digest& key_digest,
                                  std::uint32_t level) {
  if (level >= superblock.level_count) {
    throw std::out_of_range("bucket level is out of range");
  }
  const std::array<std::uint8_t, 4> level_bytes = {
      static_cast<std::uint8_t>((level >> 24U) & 0xffU),
      static_cast<std::uint8_t>((level >> 16U) & 0xffU),
      static_cast<std::uint8_t>((level >> 8U) & 0xffU),
      static_cast<std::uint8_t>(level & 0xffU)};
  auto hash = fnv1a(level_bytes.data(), level_bytes.size());
  hash = fnv1a(key_digest.data(), key_digest.size(), hash);
  const auto base = (superblock.bucket_metadata_offsets[level] -
                     superblock.bucket_metadata_offsets[0]) /
                    sizeof(BucketMetaSlot);
  return base + hash % superblock.buckets_per_level[level];
}

IndexCore::IndexCore(
    std::uintptr_t base_address, std::uint64_t region_size,
    std::uint32_t participant_id, const FormatParameters& expected,
    const std::vector<std::array<std::uint64_t, 4>>& payload_ranges,
    bool skip_payload_flush, bool memcheck_on_attach)
    : base_(reinterpret_cast<std::uint8_t*>(base_address)),
      superblock_(reinterpret_cast<Superblock*>(base_address)),
      participant_id_(participant_id),
      owner_slot_begin_(expected.owner_range(participant_id)[0]),
      owner_slot_count_(expected.owner_range(participant_id)[1]),
      skip_payload_flush_(skip_payload_flush) {
  validate_cpu_visibility_profile();
  validate_superblock(region_size, expected);
  // Logical slots remain owner-contiguous; their payload may span devices.
  // The caller fences this address-independent placement in the layout digest.
  std::uint64_t end = 0;
  for (const auto& range : payload_ranges) {
    const auto [begin, count, address, rank] = range;
    if (begin != end || count == 0 ||
        count > superblock_->payload_slot_count - end || rank >= 8 ||
        address == 0 || address % kCacheLineBytes != 0 ||
        count > (std::numeric_limits<std::uintptr_t>::max() - address) /
                    superblock_->payload_slot_bytes) {
      throw std::invalid_argument("invalid payload placement range");
    }
    payload_ranges_.push_back(
        {begin, count, address, static_cast<std::uint32_t>(rank)});
    free_slots_.resize(std::max(free_slots_.size(), std::size_t(rank + 1)));
    end += count;
  }
  if (end != superblock_->payload_slot_count) {
    throw std::invalid_argument("payload placement must cover every slot");
  }
  bit_words_per_participant_ = (superblock_->payload_slot_count + 511) / 512;
  total_bucket_count_ = (superblock_->bucket_peterson_offset -
                         superblock_->bucket_metadata_offsets[0]) /
                        sizeof(BucketMetaSlot);
  local_bucket_claimed_.assign(total_bucket_count_, false);
  local_reader_counts_.assign(superblock_->payload_slot_count, 0);
  rebuild_local_slot_state();
  // Full-index diagnostics require quiesced participants: concurrent writers
  // can expose transitional bucket/header combinations. Static contract and
  // mapping validation above always runs, even when this scan is disabled.
  if (memcheck_on_attach && !memcheck()) {
    throw std::runtime_error(
        "DAX-Coordinated L1 region requires recovery before attach");
  }
}

IndexCore::~IndexCore() {
  try {
    close();
  } catch (...) {
  }
}

void IndexCore::validate_superblock(std::uint64_t region_size,
                                    const FormatParameters& expected) {
  if (base_ == nullptr ||
      reinterpret_cast<std::uintptr_t>(base_) % kCacheLineBytes != 0 ||
      region_size < sizeof(Superblock)) {
    throw std::invalid_argument(
        "mapped metadata must contain an aligned superblock");
  }
  refresh_range(superblock_, sizeof(Superblock));
  if (superblock_->magic != kMagic) {
    throw std::runtime_error("DAX-Coordinated L1 region is not formatted");
  }
  if (superblock_->format_version != kFormatVersion) {
    throw std::runtime_error(
        "unsupported DAX-Coordinated L1 format version; stop all participants "
        "and explicitly reformat the region");
  }
  const auto canonical =
      make_superblock(expected, region_size, expected.layout());
  if (std::memcmp(superblock_, &canonical, sizeof(Superblock)) != 0) {
    throw std::runtime_error(
        "DAX-Coordinated L1 layout or digest contract mismatch");
  }
  const auto* participant =
      reinterpret_cast<const ParticipantEntry*>(
          base_ + superblock_->participant_registry_offset) +
      participant_id_;
  refresh_range(const_cast<ParticipantEntry*>(participant),
                sizeof(ParticipantEntry));
  if (participant->participant_id != participant_id_ ||
      participant->participant_state != 1 ||
      participant->owner_slot_begin != owner_slot_begin_ ||
      participant->owner_slot_count != owner_slot_count_) {
    throw std::runtime_error(
        "DAX-Coordinated L1 participant contract mismatch");
  }
}

BucketMetaSlot* IndexCore::bucket(std::uint64_t bucket_id) const {
  if (bucket_id >= total_bucket_count_) {
    throw std::out_of_range("global bucket ID is out of range");
  }
  return reinterpret_cast<BucketMetaSlot*>(
             base_ + superblock_->bucket_metadata_offsets[0]) +
         bucket_id;
}

BucketMetaSlot* IndexCore::refreshed_bucket(std::uint64_t bucket_id) const {
  auto* metadata = bucket(bucket_id);
  refresh_range(&metadata->identity, sizeof(BucketIdentityLine));
  refresh_range(&metadata->control, sizeof(BucketControlLine));
  return metadata;
}

BucketPetersonRecord* IndexCore::peterson(std::uint64_t bucket_id) const {
  if (bucket_id >= total_bucket_count_) {
    throw std::out_of_range("global bucket ID is out of range");
  }
  auto* records = reinterpret_cast<BucketPetersonRecord*>(
      base_ + superblock_->bucket_peterson_offset);
  return records + bucket_id * (superblock_->max_participants - 1);
}

PayloadSlotHeader* IndexCore::payload_header(std::uint64_t slot_id) const {
  if (slot_id >= superblock_->payload_slot_count) {
    throw std::out_of_range("global payload slot ID is out of range");
  }
  auto* headers = reinterpret_cast<PayloadSlotHeader*>(
      base_ + superblock_->payload_header_offset);
  return headers + slot_id;
}

void* IndexCore::payload_address(std::uint64_t slot_id) const {
  const auto& range = payload_range(slot_id);
  return reinterpret_cast<void*>(range.address +
                                 (slot_id - range.begin) *
                                     superblock_->payload_slot_bytes);
}

const IndexCore::PayloadRange& IndexCore::payload_range(
    std::uint64_t slot_id) const {
  for (const auto& range : payload_ranges_) {
    if (slot_id >= range.begin && slot_id - range.begin < range.count) {
      return range;
    }
  }
  throw std::out_of_range("global payload slot ID is out of range");
}

PackedBitWord* IndexCore::activity_word(std::uint32_t participant_id,
                                        std::uint64_t slot_id) const {
  auto* words = reinterpret_cast<PackedBitWord*>(
      base_ + superblock_->reader_activity_offset);
  return words + participant_id * bit_words_per_participant_ + slot_id / 512;
}

void IndexCore::rebuild_local_slot_state() {
  std::lock_guard<std::mutex> lock(local_mutex_);
  for (auto& slots : free_slots_) slots.clear();
  for (std::uint64_t slot = owner_slot_begin_;
       slot < owner_slot_begin_ + owner_slot_count_; ++slot) {
    auto* header = payload_header(slot);
    refresh_range(header, sizeof(PayloadSlotHeader));
    if (header->owner_participant_id != participant_id_) {
      throw std::runtime_error("owned payload slot header is corrupt");
    }
    if (header->allocation_state == PayloadAllocationState::kFree) {
      free_slots_[payload_range(slot).rank].push_back(slot);
    }
  }
}

bool IndexCore::claim_bucket(std::uint64_t bucket_id) {
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    if (closed_ || local_bucket_claimed_[bucket_id]) {
      return false;
    }
    local_bucket_claimed_[bucket_id] = true;
  }
  if (!try_acquire_tournament(peterson(bucket_id), participant_id_,
                              superblock_->max_participants)) {
    std::lock_guard<std::mutex> lock(local_mutex_);
    local_bucket_claimed_[bucket_id] = false;
    return false;
  }
  return true;
}

void IndexCore::release_bucket(std::uint64_t bucket_id) {
  release_tournament(peterson(bucket_id), participant_id_,
                     superblock_->max_participants);
  std::lock_guard<std::mutex> lock(local_mutex_);
  local_bucket_claimed_[bucket_id] = false;
}

IndexCore::LogicalTarget IndexCore::find_lowest_logical_target(
    const Digest& key_digest) {
  for (std::uint32_t level = 0; level < superblock_->level_count; ++level) {
    const auto bucket_id = bucket_id_for_level(*superblock_, key_digest, level);
    auto* target = refreshed_bucket(bucket_id);
    if (target->identity.key_digest != key_digest) {
      continue;
    }
    auto state = target->control.state;
    auto intent = target->control.write_intent;
    auto gate = target->control.reader_gate;
    if (!valid_state(state, intent, gate)) {
      // These fields are published together but loaded separately. A concurrent
      // writer can expose a transitional combination on a coherent host. Only
      // classify it as corruption after rechecking under the bucket claim.
      if (!claim_bucket(bucket_id)) {
        return {true, bucket_id, BucketState::kWriteLocked, WriteIntent::kNone};
      }
      target = refreshed_bucket(bucket_id);
      const auto identity = target->identity;
      state = target->control.state;
      intent = target->control.write_intent;
      gate = target->control.reader_gate;
      release_bucket(bucket_id);
      if (identity.key_digest != key_digest) continue;
      if (!valid_state(state, intent, gate)) {
        return {true, bucket_id, BucketState::kRecovery, WriteIntent::kNone};
      }
      if (state == BucketState::kFree || (state == BucketState::kWriteLocked &&
                                          intent == WriteIntent::kAllocating)) {
        continue;
      }
      return {true, bucket_id, state, intent, identity};
    }
    if (state == BucketState::kFree || (state == BucketState::kWriteLocked &&
                                        intent == WriteIntent::kAllocating)) {
      continue;
    }
    return {true, bucket_id, state, intent, target->identity};
  }
  return {};
}

ReservationResult IndexCore::make_reservation_result(
    std::uint64_t token, const BucketIdentityLine& identity) const {
  return {OperationResult::kSuccess,
          token,
          identity.global_payload_slot_id,
          identity.bucket_generation,
          identity.global_payload_slot_id * superblock_->payload_slot_bytes,
          identity.payload_length,
          identity.layout_id};
}

ReservationResult IndexCore::reserve_write(const Digest& key_digest,
                                           std::uint32_t payload_length,
                                           std::uint32_t layout_id,
                                           std::uint32_t allocation_rank) {
  if (closed_ || payload_length == 0 ||
      payload_length > superblock_->payload_slot_bytes ||
      allocation_rank >= free_slots_.size()) {
    return {OperationResult::kInvalidState};
  }
  auto target = find_lowest_logical_target(key_digest);
  if (target.found) {
    // Published keys are immutable, matching the distributed L1 contract.
    return {OperationResult::kWriterBusy};
  }
  return reserve_allocate(key_digest, payload_length, layout_id,
                          allocation_rank);
}

ReservationResult IndexCore::reserve_allocate(const Digest& key_digest,
                                              std::uint32_t payload_length,
                                              std::uint32_t layout_id,
                                              std::uint32_t allocation_rank) {
  // New allocation after key lookup reports no logical target:
  // No local payload slot -> NoLocalPayloadSlot.
  // Otherwise, probe candidate buckets starting at level 0:
  // |-- Non-FREE
  // |   |-- Same key -> WriterBusy (no retry).
  // |   `-- Other key -> next level.
  // `-- FREE -> try to claim the bucket
  //     |-- Failure -> this attempt's claims have been released
  //     |   |-- Retry budget remains -> wait 10 / 20 / 40 us, count retry
  //     |   |   `-- Repeat key lookup
  //     |   |       |-- Target found -> WriterBusy.
  //     |   |       `-- No target -> restart at level 0.
  //     |   `-- Three retries consumed -> next level without waiting.
  //     `-- Success -> recheck state under the lock
  //         |-- Non-FREE -> release lock
  //         |   |-- Same key -> WriterBusy.
  //         |   `-- Other key -> next level.
  //         `-- FREE -> reserve a local payload slot
  //             |-- None available -> release lock; NoLocalPayloadSlot.
  //             `-- Available -> publish WRITE_LOCKED and slot mapping;
  //                 return reservation with lock held. After D2H, finish_writes
  //                 publishes payload and READY, then releases the lock.
  // All levels exhausted -> WriterBusy if any claim failed, else NoFreeBucket.
  // The three-retry budget is shared across levels and is never reset here.
  auto& slots = free_slots_[allocation_rank];
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    if (slots.empty()) {
      return {OperationResult::kNoLocalPayloadSlot};
    }
  }

  // Retry contention from lookup with a budget shared across all levels, then
  // skip contended buckets. An unpublished same-key contender can therefore
  // still produce replicas at different levels. Keep waiting bounded because
  // the caller may hold a lock needed by local completion callbacks.
  constexpr std::uint32_t kMaxLockRetries = 3;
  std::uint32_t lock_retries = 0;
  bool saw_busy_bucket = false;
  for (std::uint32_t level = 0; level < superblock_->level_count;) {
    const auto bucket_id = bucket_id_for_level(*superblock_, key_digest, level);
    auto* metadata = refreshed_bucket(bucket_id);
    // Conservatively reject a same-key occupant without taking the lock;
    // skip other occupied candidates. FREE still needs rechecking after claim.
    if (metadata->control.state != BucketState::kFree) {
      if (metadata->identity.key_digest == key_digest) {
        return {OperationResult::kWriterBusy};
      }
      ++level;
      continue;
    }
    if (!claim_bucket(bucket_id)) {
      saw_busy_bucket = true;
      if (lock_retries == kMaxLockRetries) {
        ++level;
        continue;
      }
      // claim_bucket has already unwound all claims from this attempt.
      std::this_thread::sleep_for(
          std::chrono::microseconds(10U << lock_retries));
      ++lock_retries;
      if (find_lowest_logical_target(key_digest).found) {
        return {OperationResult::kWriterBusy};
      }
      level = 0;
      continue;
    }
    metadata = refreshed_bucket(bucket_id);
    const auto state = metadata->control.state;
    if (state != BucketState::kFree) {
      const bool same_key = metadata->identity.key_digest == key_digest;
      release_bucket(bucket_id);
      if (same_key) return {OperationResult::kWriterBusy};
      ++level;
      continue;
    }

    std::unique_lock<std::mutex> lock(local_mutex_);
    if (slots.empty()) {
      lock.unlock();
      release_bucket(bucket_id);
      return {OperationResult::kNoLocalPayloadSlot};
    }
    const auto slot_id = slots.front();
    slots.pop_front();
    lock.unlock();
    auto* header = payload_header(slot_id);
    refresh_range(header, sizeof(PayloadSlotHeader));
    const auto slot_generation = header->slot_generation;
    const auto bucket_generation = metadata->identity.bucket_generation + 1;

    publish_write_intent(bucket_id, WriteIntent::kAllocating);

    header->allocation_state = PayloadAllocationState::kAllocated;
    header->back_bucket_id = bucket_id;
    header->back_bucket_generation = bucket_generation;
    header->payload_length = payload_length;
    publish_range(header, sizeof(PayloadSlotHeader));

    metadata->identity.key_digest = key_digest;
    metadata->identity.bucket_generation = bucket_generation;
    metadata->identity.global_payload_slot_id = slot_id;
    metadata->identity.slot_generation = slot_generation;
    metadata->identity.payload_length = payload_length;
    metadata->identity.layout_id = layout_id;
    publish_range(&metadata->identity, sizeof(BucketIdentityLine));

    return remember_write(bucket_id);
  }

  return {saw_busy_bucket ? OperationResult::kWriterBusy
                          : OperationResult::kNoFreeBucket};
}

ReservationResult IndexCore::remember_write(std::uint64_t bucket_id) {
  const auto& identity = bucket(bucket_id)->identity;
  WriteReservation reservation{
      bucket_id, identity.global_payload_slot_id, identity.bucket_generation,
      identity.slot_generation, identity.payload_length};
  std::lock_guard<std::mutex> lock(local_mutex_);
  const auto token = next_token_++;
  write_reservations_.emplace(token, reservation);
  return make_reservation_result(token, identity);
}

bool IndexCore::mapping_is_valid(std::uint64_t bucket_id,
                                 const BucketIdentityLine& identity,
                                 const PayloadSlotHeader& header) const {
  return identity.global_payload_slot_id < superblock_->payload_slot_count &&
         header.allocation_state == PayloadAllocationState::kAllocated &&
         header.slot_generation == identity.slot_generation &&
         header.back_bucket_id == bucket_id &&
         header.back_bucket_generation == identity.bucket_generation &&
         header.payload_length == identity.payload_length;
}

void IndexCore::flush_payload(void* address, std::size_t length) const {
  if (skip_payload_flush_) {
    return;
  }
  if (superblock_->visibility_mode == kVisibilityX86ClflushoptBulkV1) {
    publish_range_bulk(address, length);
    return;
  }
  publish_range(address, length);
}

bool IndexCore::any_reader_active(std::uint64_t slot_id) const {
  const auto bit_index = slot_id & 511U;
  const auto word_index = bit_index / 64;
  const auto mask = std::uint64_t{1} << (bit_index & 63U);
  for (std::uint32_t participant = 0;
       participant < superblock_->max_participants; ++participant) {
    auto* word = activity_word(participant, slot_id);
    refresh_range(word, sizeof(PackedBitWord));
    if ((word->words[word_index] & mask) != 0) {
      return true;
    }
  }
  return false;
}

PackedBitWord* IndexCore::set_reader_activity(std::uint64_t slot_id,
                                              bool active) {
  // Caller holds local_mutex_; each participant owns its own activity bank.
  auto* word = activity_word(participant_id_, slot_id);
  auto& lane = word->words[(slot_id & 511U) / 64];
  const auto mask = std::uint64_t{1} << (slot_id & 63U);
  if (active)
    lane |= mask;
  else
    lane &= ~mask;
  return word;
}

void IndexCore::enter_local_reader(std::uint64_t slot_id, std::uint32_t count) {
  std::lock_guard<std::mutex> lock(local_mutex_);
  const auto previous = local_reader_counts_[slot_id];
  if (count == 0 ||
      previous > std::numeric_limits<std::uint32_t>::max() - count) {
    throw std::overflow_error("local reader count overflow");
  }
  local_reader_counts_[slot_id] += count;
  if (previous == 0) {
    auto* word = set_reader_activity(slot_id, true);
    publish_range(word, sizeof(PackedBitWord));
  }
}

bool IndexCore::exit_local_reader(std::uint64_t slot_id, std::uint32_t count) {
  std::lock_guard<std::mutex> lock(local_mutex_);
  if (count == 0 || local_reader_counts_[slot_id] < count) {
    return false;
  }
  local_reader_counts_[slot_id] -= count;
  if (local_reader_counts_[slot_id] == 0) {
    auto* word = set_reader_activity(slot_id, false);
    publish_range(word, sizeof(PackedBitWord));
  }
  return true;
}

OperationResult IndexCore::prepare_finish_write(std::uint64_t token,
                                                WriteReservation& reservation) {
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    const auto found = write_reservations_.find(token);
    if (found == write_reservations_.end()) {
      return OperationResult::kInvalidState;
    }
    reservation = found->second;
  }
  auto* metadata = refreshed_bucket(reservation.bucket_id);
  auto* header = payload_header(reservation.slot_id);
  refresh_range(header, sizeof(PayloadSlotHeader));
  if (metadata->identity.bucket_generation != reservation.bucket_generation ||
      metadata->identity.slot_generation != reservation.slot_generation ||
      metadata->identity.global_payload_slot_id != reservation.slot_id ||
      !mapping_is_valid(reservation.bucket_id, metadata->identity, *header) ||
      metadata->control.state != BucketState::kWriteLocked ||
      metadata->control.write_intent != WriteIntent::kAllocating) {
    fail_write(token, reservation);
    return OperationResult::kGenerationMismatch;
  }
  return OperationResult::kSuccess;
}

void IndexCore::commit_finish_write(std::uint64_t token,
                                    const WriteReservation& reservation) {
  publish_ready(reservation.bucket_id);
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    write_reservations_.erase(token);
  }
  release_bucket(reservation.bucket_id);
}

std::vector<OperationResult> IndexCore::finish_writes(
    const std::vector<std::uint64_t>& tokens) {
  std::vector<OperationResult> results(tokens.size(),
                                       OperationResult::kInvalidState);
  std::vector<WriteReservation> reservations(tokens.size());
  std::unordered_set<std::uint64_t> seen_tokens;
  seen_tokens.reserve(tokens.size());
  const bool bulk =
      superblock_->visibility_mode == kVisibilityX86ClflushoptBulkV1;
  std::vector<VisibilityRange> payload_ranges;
  payload_ranges.reserve(bulk && !skip_payload_flush_ ? tokens.size() : 0);

  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (!seen_tokens.insert(tokens[index]).second) {
      continue;
    }
    results[index] = prepare_finish_write(tokens[index], reservations[index]);
    if (results[index] != OperationResult::kSuccess || skip_payload_flush_) {
      continue;
    }
    const VisibilityRange payload{payload_address(reservations[index].slot_id),
                                  reservations[index].payload_length};
    if (bulk) {
      payload_ranges.push_back(payload);
    } else {
      publish_range(payload.address, payload.length);
    }
  }
  publish_ranges_bulk(payload_ranges.data(), payload_ranges.size());

  for (std::size_t index = 0; index < tokens.size(); ++index) {
    if (results[index] == OperationResult::kSuccess) {
      commit_finish_write(tokens[index], reservations[index]);
    }
  }
  return results;
}

void IndexCore::publish_ready(std::uint64_t bucket_id) {
  auto* metadata = bucket(bucket_id);
  metadata->control.state = BucketState::kReady;
  metadata->control.write_intent = WriteIntent::kNone;
  metadata->control.reader_gate = ReaderGate::kOpen;
  publish_range(&metadata->control, sizeof(BucketControlLine));
}

void IndexCore::publish_write_intent(std::uint64_t bucket_id,
                                     WriteIntent intent) {
  auto* metadata = bucket(bucket_id);
  metadata->control.state = BucketState::kWriteLocked;
  metadata->control.write_intent = intent;
  metadata->control.reader_gate = ReaderGate::kClosed;
  publish_range(&metadata->control, sizeof(BucketControlLine));
}

void IndexCore::fail_write(std::uint64_t token,
                           const WriteReservation& reservation) {
  auto* metadata = bucket(reservation.bucket_id);
  metadata->control.state = BucketState::kRecovery;
  metadata->control.reader_gate = ReaderGate::kClosed;
  publish_range(&metadata->control, sizeof(BucketControlLine));
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    write_reservations_.erase(token);
  }
  release_bucket(reservation.bucket_id);
}

void IndexCore::reclaim_slot(std::uint64_t bucket_id, std::uint64_t slot_id,
                             std::uint64_t bucket_generation) {
  auto* metadata = bucket(bucket_id);
  auto* header = payload_header(slot_id);
  std::memset(&metadata->identity, 0, sizeof(BucketIdentityLine));
  metadata->identity.bucket_generation = bucket_generation;
  publish_range(&metadata->identity, sizeof(BucketIdentityLine));
  metadata->control = {};
  publish_range(&metadata->control, sizeof(BucketControlLine));
  header->allocation_state = PayloadAllocationState::kFree;
  header->back_bucket_id = std::numeric_limits<std::uint64_t>::max();
  header->back_bucket_generation = 0;
  header->payload_length = 0;
  ++header->slot_generation;
  publish_range(header, sizeof(PayloadSlotHeader));
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    free_slots_[payload_range(slot_id).rank].push_back(slot_id);
  }
  release_bucket(bucket_id);
}

ReservationResult IndexCore::finish_write_and_reserve_read(
    std::uint64_t token, std::uint32_t read_count) {
  if (read_count == 0) return {OperationResult::kInvalidState};
  WriteReservation reservation{};
  const auto result = prepare_finish_write(token, reservation);
  if (result != OperationResult::kSuccess) return {result};
  auto* metadata = bucket(reservation.bucket_id);
  flush_payload(payload_address(reservation.slot_id),
                reservation.payload_length);
  publish_ready(reservation.bucket_id);
  enter_local_reader(reservation.slot_id, read_count);
  refresh_range(&metadata->identity, sizeof(BucketIdentityLine));
  refresh_range(&metadata->control, sizeof(BucketControlLine));
  if (metadata->identity.bucket_generation != reservation.bucket_generation ||
      metadata->identity.slot_generation != reservation.slot_generation ||
      metadata->control.state != BucketState::kReady ||
      metadata->control.reader_gate != ReaderGate::kOpen) {
    exit_local_reader(reservation.slot_id, read_count);
    fail_write(token, reservation);
    return {OperationResult::kGenerationMismatch};
  }
  ReadReservation read_reservation{reservation.slot_id, read_count};
  std::uint64_t read_token;
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    read_token = next_token_++;
    write_reservations_.erase(token);
    read_reservations_.emplace(read_token, read_reservation);
  }
  release_bucket(reservation.bucket_id);
  return make_reservation_result(read_token, metadata->identity);
}

std::vector<ReservationResult> IndexCore::reserve_reads(
    const std::vector<Digest>& key_digests, std::uint32_t read_count) {
  std::vector<ReservationResult> results(key_digests.size());
  if (closed_ || read_count == 0) {
    return results;
  }

  struct PreparedRead {
    std::size_t result_index;
    std::uint64_t bucket_id;
    BucketIdentityLine snapshot;
    bool active{true};
  };
  std::vector<PreparedRead> prepared;
  for (std::size_t index = 0; index < key_digests.size(); ++index) {
    const auto& digest = key_digests[index];
    const auto target = find_lowest_logical_target(digest);
    if (!target.found) {
      results[index].result = OperationResult::kNotFound;
      continue;
    }
    if (target.state == BucketState::kRecovery) {
      results[index].result = OperationResult::kRecoveryRequired;
      continue;
    }
    if (target.state == BucketState::kWriteLocked) {
      if (target.intent == WriteIntent::kInvalidating) {
        results[index].result = OperationResult::kTargetInvalidating;
      } else {
        results[index].result = OperationResult::kInvalidState;
      }
      continue;
    }

    // Lookup already supplies a READY/OPEN snapshot. Publishing reader activity
    // below and then rechecking identity/control closes the writer race.
    if (target.identity.global_payload_slot_id >=
        superblock_->payload_slot_count) {
      results[index].result = OperationResult::kCorruptForwardBackref;
      continue;
    }
    prepared.push_back({index, target.bucket_id, target.identity});
  }

  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    std::unordered_map<std::uint64_t, std::uint64_t> additions;
    for (const auto& item : prepared) {
      additions[item.snapshot.global_payload_slot_id] += read_count;
    }
    for (const auto& [slot_id, addition] : additions) {
      if (addition > std::numeric_limits<std::uint32_t>::max() ||
          local_reader_counts_[slot_id] >
              std::numeric_limits<std::uint32_t>::max() - addition) {
        throw std::overflow_error("local reader count overflow");
      }
    }
    std::unordered_set<PackedBitWord*> dirty_words;
    for (const auto& [slot_id, addition] : additions) {
      const auto previous = local_reader_counts_[slot_id];
      local_reader_counts_[slot_id] += static_cast<std::uint32_t>(addition);
      if (previous == 0) {
        dirty_words.insert(set_reader_activity(slot_id, true));
      }
    }
    publish_reader_activity(dirty_words);
  }

  for (auto& item : prepared) {
    const auto& digest = key_digests[item.result_index];
    auto* metadata = refreshed_bucket(item.bucket_id);
    const bool unchanged =
        (metadata->identity.key_digest == digest) &&
        metadata->identity.bucket_generation ==
            item.snapshot.bucket_generation &&
        metadata->identity.global_payload_slot_id ==
            item.snapshot.global_payload_slot_id &&
        metadata->identity.slot_generation == item.snapshot.slot_generation;
    const auto state = metadata->control.state;
    const auto intent = metadata->control.write_intent;
    const auto gate = metadata->control.reader_gate;
    if (!unchanged || state != BucketState::kReady ||
        gate != ReaderGate::kOpen) {
      item.active = false;
      results[item.result_index].result =
          intent == WriteIntent::kInvalidating
              ? OperationResult::kTargetInvalidating
              : OperationResult::kGenerationMismatch;
      continue;
    }
    auto* header = payload_header(item.snapshot.global_payload_slot_id);
    refresh_range(header, sizeof(PayloadSlotHeader));
    if (!mapping_is_valid(item.bucket_id, item.snapshot, *header)) {
      item.active = false;
      results[item.result_index].result =
          OperationResult::kCorruptForwardBackref;
    }
  }

  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    std::unordered_set<PackedBitWord*> dirty_words;
    for (const auto& item : prepared) {
      if (item.active) {
        continue;
      }
      const auto slot_id = item.snapshot.global_payload_slot_id;
      local_reader_counts_[slot_id] -= read_count;
      if (local_reader_counts_[slot_id] == 0) {
        dirty_words.insert(set_reader_activity(slot_id, false));
      }
    }
    publish_reader_activity(dirty_words);
  }

  for (const auto& item : prepared) {
    if (item.active) {
      flush_payload(payload_address(item.snapshot.global_payload_slot_id),
                    item.snapshot.payload_length);
    }
  }
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    for (const auto& item : prepared) {
      if (!item.active) {
        continue;
      }
      ReadReservation reservation{item.snapshot.global_payload_slot_id,
                                  read_count};
      const auto token = next_token_++;
      read_reservations_.emplace(token, reservation);
      results[item.result_index] =
          make_reservation_result(token, item.snapshot);
    }
  }
  return results;
}

std::vector<OperationResult> IndexCore::finish_reads(
    const std::vector<std::uint64_t>& tokens,
    const std::vector<std::uint32_t>& read_counts) {
  if (tokens.size() != read_counts.size()) {
    throw std::invalid_argument("read token and count vectors must match");
  }
  std::vector<OperationResult> results(tokens.size(),
                                       OperationResult::kInvalidState);
  std::unordered_set<PackedBitWord*> dirty_activity;
  std::lock_guard<std::mutex> lock(local_mutex_);
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const auto count = read_counts[index];
    const auto found = read_reservations_.find(tokens[index]);
    if (found == read_reservations_.end() || count == 0 ||
        count > found->second.count) {
      continue;
    }
    const auto reservation = found->second;
    if (local_reader_counts_[reservation.slot_id] < count) {
      continue;
    }
    found->second.count -= count;
    if (found->second.count == 0) {
      read_reservations_.erase(found);
    }
    local_reader_counts_[reservation.slot_id] -= count;

    if (local_reader_counts_[reservation.slot_id] == 0) {
      dirty_activity.insert(set_reader_activity(reservation.slot_id, false));
    }
    results[index] = OperationResult::kSuccess;
  }
  publish_reader_activity(dirty_activity);
  return results;
}

ReservationResult IndexCore::delete_key(const Digest& key_digest) {
  const auto target = find_lowest_logical_target(key_digest);
  if (!target.found) {
    return {OperationResult::kNotFound};
  }
  if (target.state != BucketState::kReady) {
    return {OperationResult::kInvalidState};
  }
  auto* metadata = bucket(target.bucket_id);
  refresh_range(&metadata->identity, sizeof(BucketIdentityLine));
  const auto snapshot = metadata->identity;
  const auto slot_id = snapshot.global_payload_slot_id;
  auto* header = payload_header(slot_id);
  refresh_range(header, sizeof(PayloadSlotHeader));
  if (header->owner_participant_id != participant_id_ ||
      slot_id < owner_slot_begin_ ||
      slot_id >= owner_slot_begin_ + owner_slot_count_) {
    return {OperationResult::kOwnerMismatch};
  }
  if (header->slot_generation != snapshot.slot_generation ||
      header->back_bucket_id != target.bucket_id ||
      header->back_bucket_generation != snapshot.bucket_generation) {
    return {OperationResult::kNotFound};
  }
  // Avoid taking the writer lock or closing admission for an active reader.
  // Readers can still enter, so recheck after closing the gate below.
  if (any_reader_active(slot_id)) {
    return {OperationResult::kActiveReader};
  }
  if (!claim_bucket(target.bucket_id)) {
    return {OperationResult::kWriterBusy};
  }
  metadata = refreshed_bucket(target.bucket_id);
  refresh_range(header, sizeof(PayloadSlotHeader));
  if (!mapping_is_valid(target.bucket_id, metadata->identity, *header) ||
      metadata->identity.slot_generation != snapshot.slot_generation ||
      metadata->identity.bucket_generation != snapshot.bucket_generation ||
      metadata->identity.key_digest != key_digest) {
    release_bucket(target.bucket_id);
    return {OperationResult::kNotFound};
  }
  if (metadata->control.state != BucketState::kReady) {
    release_bucket(target.bucket_id);
    return {OperationResult::kInvalidState};
  }

  publish_write_intent(target.bucket_id, WriteIntent::kInvalidating);

  if (any_reader_active(slot_id)) {
    publish_ready(target.bucket_id);
    release_bucket(target.bucket_id);
    return {OperationResult::kActiveReader};
  }

  reclaim_slot(target.bucket_id, slot_id, metadata->identity.bucket_generation);
  return make_reservation_result(0, snapshot);
}

CoreStatus IndexCore::report_status() {
  CoreStatus status{};
  std::lock_guard<std::mutex> lock(local_mutex_);
  for (const auto& slots : free_slots_) {
    status.free_slots_by_rank.push_back(slots.size());
    status.free_slots += slots.size();
  }
  status.used_slots = owner_slot_count_ - status.free_slots;
  status.active_read_reservations = read_reservations_.size();
  status.active_write_reservations = write_reservations_.size();
  return status;
}

bool IndexCore::memcheck() {
  for (std::uint64_t bucket_id = 0; bucket_id < total_bucket_count_;
       ++bucket_id) {
    auto* metadata = refreshed_bucket(bucket_id);
    const auto state = metadata->control.state;
    const auto intent = metadata->control.write_intent;
    const auto gate = metadata->control.reader_gate;
    if (!valid_state(state, intent, gate) || state == BucketState::kRecovery) {
      return false;
    }
    if (state == BucketState::kReady || state == BucketState::kWriteLocked) {
      if (metadata->identity.global_payload_slot_id >=
          superblock_->payload_slot_count) {
        return false;
      }
      auto* header = payload_header(metadata->identity.global_payload_slot_id);
      refresh_range(header, sizeof(PayloadSlotHeader));
      if (!mapping_is_valid(bucket_id, metadata->identity, *header)) {
        return false;
      }
    }
  }
  return true;
}

void IndexCore::close() {
  std::vector<std::uint64_t> read_tokens;
  std::vector<std::uint32_t> read_counts;
  {
    std::lock_guard<std::mutex> lock(local_mutex_);
    if (closed_) {
      return;
    }
    for (const auto& [token, reservation] : read_reservations_) {
      read_tokens.push_back(token);
      read_counts.push_back(reservation.count);
    }
  }
  // Pop reservations directly: shutdown is the only write-cancellation path.
  while (true) {
    WriteReservation reservation;
    {
      std::lock_guard<std::mutex> lock(local_mutex_);
      if (write_reservations_.empty()) break;
      const auto first = write_reservations_.begin();
      reservation = first->second;
      write_reservations_.erase(first);
    }
    reclaim_slot(reservation.bucket_id, reservation.slot_id,
                 reservation.bucket_generation);
  }
  finish_reads(read_tokens, read_counts);
  std::lock_guard<std::mutex> lock(local_mutex_);
  closed_ = true;
}

}  // namespace lmcache::dax_coordinated_l1
