// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "dax_coordinated_l1/layout.h"

namespace lmcache::dax_coordinated_l1 {

enum class OperationResult : std::uint8_t {
  kSuccess = 0,
  kNotFound,
  kTargetInvalidating = 3,
  kWriterBusy,
  kNoFreeBucket,
  kNoLocalPayloadSlot,
  kGenerationMismatch,
  kActiveReader,
  kOwnerMismatch,
  kInvalidState,
  kCorruptForwardBackref = 12,
  kRecoveryRequired,
};

struct ReservationResult {
  OperationResult result{OperationResult::kInvalidState};
  std::uint64_t token{0};
  std::uint64_t global_payload_slot_id{0};
  std::uint64_t bucket_generation{0};
  std::uint64_t payload_offset{0};
  std::uint32_t payload_length{0};
  std::uint32_t layout_id{0};
};

struct CoreStatus {
  std::vector<std::uint64_t> free_slots_by_rank;
  std::uint64_t free_slots{0};
  std::uint64_t used_slots{0};
  std::uint64_t active_read_reservations{0};
  std::uint64_t active_write_reservations{0};
};

struct FormatParameters {
  std::uint64_t region_epoch{0};
  std::vector<std::uint64_t> buckets_per_level;
  std::uint64_t payload_slot_bytes{0};
  std::uint64_t payload_slot_count{0};
  std::uint64_t payload_alignment{0};
  std::uint32_t visibility_mode{kVisibilityX86Clflush64BV1};
  std::uint64_t payload_region_size{0};
  std::uint64_t participant_0_slot_count{0};
  std::uint32_t participant_count{2};
  Digest layout_profile_digest{};
  Digest hardware_qualification_digest{};
  Digest region_id_digest{};

  // Participant zero owns the prefix; peers equally divide the remainder.
  std::array<std::uint64_t, 2> owner_range(std::uint32_t participant) const {
    if ((participant_count != 2 && participant_count != 4) ||
        participant >= participant_count ||
        participant_0_slot_count > payload_slot_count ||
        (payload_slot_count - participant_0_slot_count) %
                (participant_count - 1) !=
            0) {
      throw std::invalid_argument("invalid participant ownership boundary");
    }
    const auto peer_count = (payload_slot_count - participant_0_slot_count) /
                            (participant_count - 1);
    return participant == 0
               ? std::array<std::uint64_t, 2>{0, participant_0_slot_count}
               : std::array<std::uint64_t, 2>{
                     participant_0_slot_count + (participant - 1) * peer_count,
                     peer_count};
  }

  LayoutDescription layout() const {
    return calculate_layout(buckets_per_level, payload_slot_count,
                            payload_slot_bytes, payload_alignment,
                            participant_count);
  }
};

void format_region(void* base, std::uint64_t region_size,
                   const FormatParameters& parameters);

std::uint64_t bucket_id_for_level(const Superblock& superblock,
                                  const Digest& key_digest,
                                  std::uint32_t level);

class IndexCore {
 public:
  IndexCore(std::uintptr_t base_address, std::uint64_t region_size,
            std::uint32_t participant_id, const FormatParameters& expected,
            const std::vector<std::array<std::uint64_t, 4>>& payload_ranges,
            bool skip_payload_flush = false, bool memcheck_on_attach = false);
  ~IndexCore();

  IndexCore(const IndexCore&) = delete;
  IndexCore& operator=(const IndexCore&) = delete;

  ReservationResult reserve_write(const Digest& key_digest,
                                  std::uint32_t payload_length,
                                  std::uint32_t layout_id,
                                  std::uint32_t allocation_rank = 0);
  std::vector<OperationResult> finish_writes(
      const std::vector<std::uint64_t>& tokens);
  ReservationResult finish_write_and_reserve_read(std::uint64_t token,
                                                  std::uint32_t read_count);

  std::vector<ReservationResult> reserve_reads(
      const std::vector<Digest>& key_digests, std::uint32_t read_count);
  std::vector<OperationResult> finish_reads(
      const std::vector<std::uint64_t>& tokens,
      const std::vector<std::uint32_t>& read_counts);

  ReservationResult delete_key(const Digest& key_digest);

  // Snapshot owner-local payload slots and reservations without scanning
  // buckets.
  CoreStatus report_status();
  bool memcheck();
  void close();

 private:
  struct PayloadRange {
    std::uint64_t begin;
    std::uint64_t count;
    std::uintptr_t address;
    std::uint32_t rank;
  };
  struct LogicalTarget {
    bool found{false};
    std::uint64_t bucket_id{0};
    BucketState state{BucketState::kFree};
    WriteIntent intent{WriteIntent::kNone};
    BucketIdentityLine identity{};
  };

  struct WriteReservation {
    std::uint64_t bucket_id{0};
    std::uint64_t slot_id{0};
    std::uint64_t bucket_generation{0};
    std::uint64_t slot_generation{0};
    std::uint32_t payload_length{0};
  };

  struct ReadReservation {
    std::uint64_t slot_id{0};
    std::uint32_t count{0};
  };

  LogicalTarget find_lowest_logical_target(const Digest& key_digest);
  ReservationResult reserve_allocate(const Digest& key_digest,
                                     std::uint32_t payload_length,
                                     std::uint32_t layout_id,
                                     std::uint32_t allocation_rank);
  ReservationResult make_reservation_result(
      std::uint64_t token, const BucketIdentityLine& identity) const;

  BucketMetaSlot* bucket(std::uint64_t bucket_id) const;
  BucketMetaSlot* refreshed_bucket(std::uint64_t bucket_id) const;
  BucketPetersonRecord* peterson(std::uint64_t bucket_id) const;
  PayloadSlotHeader* payload_header(std::uint64_t slot_id) const;
  void* payload_address(std::uint64_t slot_id) const;
  const PayloadRange& payload_range(std::uint64_t slot_id) const;
  PackedBitWord* activity_word(std::uint32_t participant_id,
                               std::uint64_t slot_id) const;
  bool claim_bucket(std::uint64_t bucket_id);
  void release_bucket(std::uint64_t bucket_id);
  bool any_reader_active(std::uint64_t slot_id) const;
  void enter_local_reader(std::uint64_t slot_id, std::uint32_t count);
  bool exit_local_reader(std::uint64_t slot_id, std::uint32_t count);
  PackedBitWord* set_reader_activity(std::uint64_t slot_id, bool active);
  bool mapping_is_valid(std::uint64_t bucket_id,
                        const BucketIdentityLine& identity,
                        const PayloadSlotHeader& header) const;
  // Both qualified profiles use the same payload flush for PUT and GET.
  void flush_payload(void* address, std::size_t length) const;
  ReservationResult remember_write(std::uint64_t bucket_id);
  OperationResult prepare_finish_write(std::uint64_t token,
                                       WriteReservation& reservation);
  void commit_finish_write(std::uint64_t token,
                           const WriteReservation& reservation);
  void publish_ready(std::uint64_t bucket_id);
  void publish_write_intent(std::uint64_t bucket_id, WriteIntent intent);
  void fail_write(std::uint64_t token, const WriteReservation& reservation);
  void reclaim_slot(std::uint64_t bucket_id, std::uint64_t slot_id,
                    std::uint64_t bucket_generation);
  void rebuild_local_slot_state();
  void validate_superblock(std::uint64_t region_size,
                           const FormatParameters& expected);

  std::uint8_t* base_;
  Superblock* superblock_;
  std::uint32_t participant_id_;
  std::uint64_t owner_slot_begin_;
  std::uint64_t owner_slot_count_;
  std::uint64_t bit_words_per_participant_;
  std::uint64_t total_bucket_count_{0};
  bool skip_payload_flush_{false};
  bool closed_{false};

  mutable std::mutex local_mutex_;
  std::vector<bool> local_bucket_claimed_;
  std::vector<std::uint32_t> local_reader_counts_;
  std::vector<PayloadRange> payload_ranges_;
  std::vector<std::deque<std::uint64_t>> free_slots_;
  std::unordered_map<std::uint64_t, WriteReservation> write_reservations_;
  std::unordered_map<std::uint64_t, ReadReservation> read_reservations_;
  std::uint64_t next_token_{1};
};

}  // namespace lmcache::dax_coordinated_l1
