// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lmcache::mp::msgpack {

using Bytes = std::vector<std::uint8_t>;

struct DecodedValue {
  enum class Kind : std::uint8_t {
    kNil,
    kString,
    kUnsigned,
    kUnsignedArray,
    kBool,
  };

  Kind kind = Kind::kNil;
  std::string string_value;
  std::uint64_t unsigned_value = 0;
  std::vector<std::uint64_t> unsigned_array_value;
  bool bool_value = false;
};

struct DecodedBlockAllocationRecord {
  std::string req_id;
  std::vector<std::uint64_t> new_block_ids;
  std::vector<std::uint64_t> new_token_ids;
};

struct DecodedCbMatchResult {
  std::uint64_t old_st = 0;
  std::uint64_t old_ed = 0;
  std::uint64_t cur_st = 0;
  std::uint64_t cur_ed = 0;
  Bytes hash;
};

struct DecodedCudaIpcWrapper {
  std::string kind;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint64_t> stride;
  std::uint64_t storage_offset = 0;
  std::string device_uuid;
  std::uint64_t raw_nbytes = 0;
  Bytes ipc_handle;
  std::uint64_t ipc_handle_bytes = 0;
  std::uint64_t storage_bytes = 0;
  std::uint64_t storage_offset_bytes = 0;
  Bytes ref_counter_handle;
  std::uint64_t ref_counter_handle_bytes = 0;
  std::uint64_t ref_counter_offset = 0;
  Bytes event_handle;
  std::uint64_t event_handle_bytes = 0;
  bool event_sync_required = false;
};

std::optional<std::uint64_t> DecodeUnsigned(const Bytes& bytes);
std::optional<std::size_t> DecodeArraySizeHeader(const Bytes& bytes);
std::optional<std::vector<std::uint64_t>> DecodeUnsignedArray(
    const Bytes& bytes);
std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>>
DecodeUnsignedPairArray(const Bytes& bytes);
std::optional<std::string> DecodeString(const Bytes& bytes);
std::optional<Bytes> DecodeBinary(const Bytes& bytes);
std::optional<std::unordered_map<std::string, DecodedValue>> DecodeStringMap(
    const Bytes& bytes);
std::optional<std::vector<DecodedBlockAllocationRecord>>
DecodeBlockAllocationRecords(const Bytes& bytes);
std::optional<std::vector<DecodedCbMatchResult>> DecodeCbMatchResults(
    const Bytes& bytes);
std::optional<std::vector<DecodedCudaIpcWrapper>> DecodeCudaIpcWrapperArray(
    const Bytes& bytes);

Bytes EncodeNil();
Bytes EncodeBool(bool value);
Bytes EncodeUnsigned(std::uint64_t value);
Bytes EncodeEmptyArray();
Bytes EncodeString(const std::string& value);
Bytes EncodeBytes(const Bytes& value);
Bytes EncodeBytesBoolTuple(const Bytes& bytes, bool value);

}  // namespace lmcache::mp::msgpack
