// SPDX-License-Identifier: Apache-2.0
#include "lmcache_mp_cpp/msgpack_lite.h"

#include <limits>
#include <utility>

namespace lmcache::mp::msgpack {
namespace {

void AppendU16(Bytes& out, std::uint16_t value) {
  out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
  out.push_back(static_cast<std::uint8_t>(value & 0xff));
}

void AppendU32(Bytes& out, std::uint32_t value) {
  out.push_back(static_cast<std::uint8_t>((value >> 24) & 0xff));
  out.push_back(static_cast<std::uint8_t>((value >> 16) & 0xff));
  out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xff));
  out.push_back(static_cast<std::uint8_t>(value & 0xff));
}

void AppendU64(Bytes& out, std::uint64_t value) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    out.push_back(static_cast<std::uint8_t>((value >> shift) & 0xff));
  }
}

std::optional<std::uint16_t> ReadU16(const Bytes& bytes, std::size_t offset) {
  if (bytes.size() < offset + 2) {
    return std::nullopt;
  }
  return static_cast<std::uint16_t>((bytes[offset] << 8) | bytes[offset + 1]);
}

std::optional<std::uint32_t> ReadU32(const Bytes& bytes, std::size_t offset) {
  if (bytes.size() < offset + 4) {
    return std::nullopt;
  }
  return (static_cast<std::uint32_t>(bytes[offset]) << 24) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 16) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 8) |
         static_cast<std::uint32_t>(bytes[offset + 3]);
}

std::optional<std::uint64_t> ReadU64(const Bytes& bytes, std::size_t offset) {
  if (bytes.size() < offset + 8) {
    return std::nullopt;
  }
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    value = (value << 8) | bytes[offset + i];
  }
  return value;
}

std::optional<std::size_t> DecodeArrayHeader(const Bytes& bytes,
                                             std::size_t* offset) {
  if (*offset >= bytes.size()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[(*offset)++];
  if ((tag & 0xf0) == 0x90) {
    return tag & 0x0f;
  }
  if (tag == 0xdc) {
    auto maybe_len = ReadU16(bytes, *offset);
    if (!maybe_len) {
      return std::nullopt;
    }
    *offset += 2;
    return static_cast<std::size_t>(*maybe_len);
  }
  if (tag == 0xdd) {
    auto maybe_len = ReadU32(bytes, *offset);
    if (!maybe_len) {
      return std::nullopt;
    }
    *offset += 4;
    return static_cast<std::size_t>(*maybe_len);
  }
  return std::nullopt;
}

std::optional<std::size_t> DecodeMapHeader(const Bytes& bytes,
                                           std::size_t* offset) {
  if (*offset >= bytes.size()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[(*offset)++];
  if ((tag & 0xf0) == 0x80) {
    return tag & 0x0f;
  }
  if (tag == 0xde) {
    auto maybe_len = ReadU16(bytes, *offset);
    if (!maybe_len) {
      return std::nullopt;
    }
    *offset += 2;
    return static_cast<std::size_t>(*maybe_len);
  }
  if (tag == 0xdf) {
    auto maybe_len = ReadU32(bytes, *offset);
    if (!maybe_len) {
      return std::nullopt;
    }
    *offset += 4;
    return static_cast<std::size_t>(*maybe_len);
  }
  return std::nullopt;
}

std::optional<std::uint64_t> DecodeUnsignedAt(const Bytes& bytes,
                                              std::size_t* offset) {
  if (*offset >= bytes.size()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[(*offset)++];
  if (tag <= 0x7f) {
    return tag;
  }
  if (tag == 0xcc && *offset + 1 <= bytes.size()) {
    return bytes[(*offset)++];
  }
  if (tag == 0xcd) {
    auto value = ReadU16(bytes, *offset);
    if (!value) {
      return std::nullopt;
    }
    *offset += 2;
    return *value;
  }
  if (tag == 0xce) {
    auto value = ReadU32(bytes, *offset);
    if (!value) {
      return std::nullopt;
    }
    *offset += 4;
    return *value;
  }
  if (tag == 0xcf) {
    auto value = ReadU64(bytes, *offset);
    if (!value) {
      return std::nullopt;
    }
    *offset += 8;
    return *value;
  }
  return std::nullopt;
}

std::optional<bool> DecodeBoolAt(const Bytes& bytes, std::size_t* offset) {
  if (*offset >= bytes.size()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[(*offset)++];
  if (tag == 0xc2) {
    return false;
  }
  if (tag == 0xc3) {
    return true;
  }
  return std::nullopt;
}

std::optional<std::string> DecodeStringAt(const Bytes& bytes,
                                          std::size_t* offset) {
  if (*offset >= bytes.size()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[(*offset)++];
  std::size_t len = 0;
  if ((tag & 0xe0) == 0xa0) {
    len = tag & 0x1f;
  } else if (tag == 0xd9 && *offset < bytes.size()) {
    len = bytes[(*offset)++];
  } else if (tag == 0xda) {
    auto maybe_len = ReadU16(bytes, *offset);
    if (!maybe_len) {
      return std::nullopt;
    }
    len = *maybe_len;
    *offset += 2;
  } else if (tag == 0xdb) {
    auto maybe_len = ReadU32(bytes, *offset);
    if (!maybe_len) {
      return std::nullopt;
    }
    len = *maybe_len;
    *offset += 4;
  } else {
    return std::nullopt;
  }
  if (bytes.size() < *offset + len) {
    return std::nullopt;
  }
  std::string out(bytes.begin() + static_cast<std::ptrdiff_t>(*offset),
                  bytes.begin() + static_cast<std::ptrdiff_t>(*offset + len));
  *offset += len;
  return out;
}

std::optional<Bytes> DecodeBinaryAt(const Bytes& bytes, std::size_t* offset) {
  if (*offset >= bytes.size()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[(*offset)++];
  std::size_t len = 0;
  if (tag == 0xc4 && *offset < bytes.size()) {
    len = bytes[(*offset)++];
  } else if (tag == 0xc5) {
    auto maybe_len = ReadU16(bytes, *offset);
    if (!maybe_len) {
      return std::nullopt;
    }
    len = *maybe_len;
    *offset += 2;
  } else if (tag == 0xc6) {
    auto maybe_len = ReadU32(bytes, *offset);
    if (!maybe_len) {
      return std::nullopt;
    }
    len = *maybe_len;
    *offset += 4;
  } else {
    return std::nullopt;
  }
  if (bytes.size() < *offset + len) {
    return std::nullopt;
  }
  Bytes out(bytes.begin() + static_cast<std::ptrdiff_t>(*offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(*offset + len));
  *offset += len;
  return out;
}

std::optional<std::vector<std::uint64_t>> DecodeUnsignedArrayAt(
    const Bytes& bytes, std::size_t* offset) {
  const auto maybe_len = DecodeArrayHeader(bytes, offset);
  if (!maybe_len) {
    return std::nullopt;
  }
  std::vector<std::uint64_t> out;
  out.reserve(*maybe_len);
  for (std::size_t i = 0; i < *maybe_len; ++i) {
    auto value = DecodeUnsignedAt(bytes, offset);
    if (!value) {
      return std::nullopt;
    }
    out.push_back(*value);
  }
  return out;
}

std::optional<std::pair<std::uint64_t, std::uint64_t>> DecodeUnsignedPairAt(
    const Bytes& bytes, std::size_t* offset) {
  const auto maybe_len = DecodeArrayHeader(bytes, offset);
  if (!maybe_len || *maybe_len != 2) {
    return std::nullopt;
  }
  auto first = DecodeUnsignedAt(bytes, offset);
  auto second = DecodeUnsignedAt(bytes, offset);
  if (!first || !second || *first > *second) {
    return std::nullopt;
  }
  return std::make_pair(*first, *second);
}

bool SkipValueAt(const Bytes& bytes, std::size_t* offset);

bool SkipBytes(const Bytes& bytes, std::size_t* offset, std::uint64_t len) {
  if (len > std::numeric_limits<std::size_t>::max() ||
      bytes.size() < *offset + static_cast<std::size_t>(len)) {
    return false;
  }
  *offset += static_cast<std::size_t>(len);
  return true;
}

bool SkipN(const Bytes& bytes, std::size_t* offset, std::uint64_t count) {
  for (std::uint64_t i = 0; i < count; ++i) {
    if (!SkipValueAt(bytes, offset)) {
      return false;
    }
  }
  return true;
}

bool SkipValueAt(const Bytes& bytes, std::size_t* offset) {
  if (*offset >= bytes.size()) {
    return false;
  }
  const std::uint8_t tag = bytes[(*offset)++];
  if (tag <= 0x7f || tag >= 0xe0 || tag == 0xc0 || tag == 0xc2 || tag == 0xc3) {
    return true;
  }
  if ((tag & 0xe0) == 0xa0) {
    return SkipBytes(bytes, offset, tag & 0x1f);
  }
  if ((tag & 0xf0) == 0x90) {
    return SkipN(bytes, offset, tag & 0x0f);
  }
  if ((tag & 0xf0) == 0x80) {
    return SkipN(bytes, offset, 2 * (tag & 0x0f));
  }

  std::uint64_t len = 0;
  switch (tag) {
    case 0xc4:
    case 0xd9:
      if (*offset >= bytes.size()) {
        return false;
      }
      len = bytes[(*offset)++];
      return SkipBytes(bytes, offset, len);
    case 0xc5: {
      auto value = ReadU16(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 2;
      return SkipBytes(bytes, offset, *value);
    }
    case 0xc6: {
      auto value = ReadU32(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 4;
      return SkipBytes(bytes, offset, *value);
    }
    case 0xda: {
      auto value = ReadU16(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 2;
      return SkipBytes(bytes, offset, *value);
    }
    case 0xdb: {
      auto value = ReadU32(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 4;
      return SkipBytes(bytes, offset, *value);
    }
    case 0xdc: {
      auto value = ReadU16(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 2;
      return SkipN(bytes, offset, *value);
    }
    case 0xdd: {
      auto value = ReadU32(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 4;
      return SkipN(bytes, offset, *value);
    }
    case 0xde: {
      auto value = ReadU16(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 2;
      return SkipN(bytes, offset, 2 * (*value));
    }
    case 0xdf: {
      auto value = ReadU32(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 4;
      return SkipN(bytes, offset, 2 * (*value));
    }
    case 0xca:
      return SkipBytes(bytes, offset, 4);
    case 0xcb:
      return SkipBytes(bytes, offset, 8);
    case 0xcc:
    case 0xd0:
      return SkipBytes(bytes, offset, 1);
    case 0xcd:
    case 0xd1:
      return SkipBytes(bytes, offset, 2);
    case 0xce:
    case 0xd2:
      return SkipBytes(bytes, offset, 4);
    case 0xcf:
    case 0xd3:
      return SkipBytes(bytes, offset, 8);
    case 0xd4:
      return SkipBytes(bytes, offset, 2);
    case 0xd5:
      return SkipBytes(bytes, offset, 3);
    case 0xd6:
      return SkipBytes(bytes, offset, 5);
    case 0xd7:
      return SkipBytes(bytes, offset, 9);
    case 0xd8:
      return SkipBytes(bytes, offset, 17);
    case 0xc7:
      if (*offset >= bytes.size()) {
        return false;
      }
      len = bytes[(*offset)++];
      return SkipBytes(bytes, offset, 1 + len);
    case 0xc8: {
      auto value = ReadU16(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 2;
      return SkipBytes(bytes, offset, 1 + *value);
    }
    case 0xc9: {
      auto value = ReadU32(bytes, *offset);
      if (!value) {
        return false;
      }
      *offset += 4;
      return SkipBytes(bytes, offset, 1 + *value);
    }
    default:
      return false;
  }
}

std::optional<std::pair<std::int8_t, Bytes>> DecodeExtAt(const Bytes& bytes,
                                                         std::size_t* offset) {
  if (*offset >= bytes.size()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[(*offset)++];
  std::uint64_t len = 0;
  if (tag == 0xd4) {
    len = 1;
  } else if (tag == 0xd5) {
    len = 2;
  } else if (tag == 0xd6) {
    len = 4;
  } else if (tag == 0xd7) {
    len = 8;
  } else if (tag == 0xd8) {
    len = 16;
  } else if (tag == 0xc7) {
    if (*offset >= bytes.size()) {
      return std::nullopt;
    }
    len = bytes[(*offset)++];
  } else if (tag == 0xc8) {
    auto value = ReadU16(bytes, *offset);
    if (!value) {
      return std::nullopt;
    }
    *offset += 2;
    len = *value;
  } else if (tag == 0xc9) {
    auto value = ReadU32(bytes, *offset);
    if (!value) {
      return std::nullopt;
    }
    *offset += 4;
    len = *value;
  } else {
    return std::nullopt;
  }
  if (*offset >= bytes.size()) {
    return std::nullopt;
  }
  const std::int8_t code = static_cast<std::int8_t>(bytes[(*offset)++]);
  if (len > std::numeric_limits<std::size_t>::max() ||
      bytes.size() < *offset + static_cast<std::size_t>(len)) {
    return std::nullopt;
  }
  Bytes data(bytes.begin() + static_cast<std::ptrdiff_t>(*offset),
             bytes.begin() + static_cast<std::ptrdiff_t>(
                                 *offset + static_cast<std::size_t>(len)));
  *offset += static_cast<std::size_t>(len);
  return std::make_pair(code, std::move(data));
}

bool DecodeCudaHandleAt(const Bytes& bytes, std::size_t* offset,
                        DecodedCudaIpcWrapper* wrapper) {
  const auto maybe_len = DecodeArrayHeader(bytes, offset);
  if (!maybe_len || *maybe_len != 8) {
    return false;
  }
  auto device_index = DecodeUnsignedAt(bytes, offset);
  (void)device_index;
  auto handle = DecodeBinaryAt(bytes, offset);
  auto storage_bytes = DecodeUnsignedAt(bytes, offset);
  auto storage_offset_bytes = DecodeUnsignedAt(bytes, offset);
  auto ref_counter_handle = DecodeBinaryAt(bytes, offset);
  auto ref_counter_offset = DecodeUnsignedAt(bytes, offset);
  auto event_handle = DecodeBinaryAt(bytes, offset);
  auto event_sync_required = DecodeBoolAt(bytes, offset);
  if (!handle || !storage_bytes || !storage_offset_bytes ||
      !ref_counter_handle || !ref_counter_offset || !event_handle ||
      !event_sync_required) {
    return false;
  }
  wrapper->ipc_handle_bytes = handle->size();
  wrapper->ref_counter_handle_bytes = ref_counter_handle->size();
  wrapper->event_handle_bytes = event_handle->size();
  wrapper->ipc_handle = std::move(*handle);
  wrapper->storage_bytes = *storage_bytes;
  wrapper->storage_offset_bytes = *storage_offset_bytes;
  wrapper->ref_counter_handle = std::move(*ref_counter_handle);
  wrapper->ref_counter_offset = *ref_counter_offset;
  wrapper->event_handle = std::move(*event_handle);
  wrapper->event_sync_required = *event_sync_required;
  return true;
}

}  // namespace

std::optional<std::uint64_t> DecodeUnsigned(const Bytes& bytes) {
  if (bytes.empty()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[0];
  if (tag <= 0x7f) {
    return tag;
  }
  if (tag == 0xcc && bytes.size() == 2) {
    return bytes[1];
  }
  if (tag == 0xcd && bytes.size() == 3) {
    return ReadU16(bytes, 1);
  }
  if (tag == 0xce && bytes.size() == 5) {
    return ReadU32(bytes, 1);
  }
  if (tag == 0xcf && bytes.size() == 9) {
    return ReadU64(bytes, 1);
  }
  return std::nullopt;
}

std::optional<std::size_t> DecodeArraySizeHeader(const Bytes& bytes) {
  std::size_t offset = 0;
  return DecodeArrayHeader(bytes, &offset);
}

std::optional<std::vector<std::uint64_t>> DecodeUnsignedArray(
    const Bytes& bytes) {
  std::size_t offset = 0;
  auto out = DecodeUnsignedArrayAt(bytes, &offset);
  if (!out) {
    return std::nullopt;
  }
  if (offset != bytes.size()) {
    return std::nullopt;
  }
  return out;
}

std::optional<std::vector<std::pair<std::uint64_t, std::uint64_t>>>
DecodeUnsignedPairArray(const Bytes& bytes) {
  std::size_t offset = 0;
  const auto maybe_len = DecodeArrayHeader(bytes, &offset);
  if (!maybe_len) {
    return std::nullopt;
  }
  std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
  out.reserve(*maybe_len);
  for (std::size_t i = 0; i < *maybe_len; ++i) {
    auto pair = DecodeUnsignedPairAt(bytes, &offset);
    if (!pair) {
      return std::nullopt;
    }
    out.push_back(*pair);
  }
  if (offset != bytes.size()) {
    return std::nullopt;
  }
  return out;
}

std::optional<std::string> DecodeString(const Bytes& bytes) {
  if (bytes.empty()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[0];
  std::size_t offset = 1;
  std::size_t len = 0;
  if ((tag & 0xe0) == 0xa0) {
    len = tag & 0x1f;
  } else if (tag == 0xd9 && bytes.size() >= 2) {
    len = bytes[1];
    offset = 2;
  } else if (tag == 0xda) {
    auto maybe_len = ReadU16(bytes, 1);
    if (!maybe_len) {
      return std::nullopt;
    }
    len = *maybe_len;
    offset = 3;
  } else if (tag == 0xdb) {
    auto maybe_len = ReadU32(bytes, 1);
    if (!maybe_len) {
      return std::nullopt;
    }
    len = *maybe_len;
    offset = 5;
  } else {
    return std::nullopt;
  }
  if (bytes.size() != offset + len) {
    return std::nullopt;
  }
  return std::string(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                     bytes.end());
}

std::optional<Bytes> DecodeBinary(const Bytes& bytes) {
  if (bytes.empty()) {
    return std::nullopt;
  }
  const std::uint8_t tag = bytes[0];
  std::size_t offset = 1;
  std::size_t len = 0;
  if (tag == 0xc4 && bytes.size() >= 2) {
    len = bytes[1];
    offset = 2;
  } else if (tag == 0xc5) {
    auto maybe_len = ReadU16(bytes, 1);
    if (!maybe_len) {
      return std::nullopt;
    }
    len = *maybe_len;
    offset = 3;
  } else if (tag == 0xc6) {
    auto maybe_len = ReadU32(bytes, 1);
    if (!maybe_len) {
      return std::nullopt;
    }
    len = *maybe_len;
    offset = 5;
  } else {
    return std::nullopt;
  }
  if (bytes.size() != offset + len) {
    return std::nullopt;
  }
  return Bytes(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
               bytes.end());
}

std::optional<std::unordered_map<std::string, DecodedValue>> DecodeStringMap(
    const Bytes& bytes) {
  std::size_t offset = 0;
  const auto maybe_len = DecodeMapHeader(bytes, &offset);
  if (!maybe_len) {
    return std::nullopt;
  }

  std::unordered_map<std::string, DecodedValue> out;
  for (std::size_t i = 0; i < *maybe_len; ++i) {
    auto key = DecodeStringAt(bytes, &offset);
    if (!key || offset >= bytes.size()) {
      return std::nullopt;
    }

    DecodedValue value;
    const std::uint8_t tag = bytes[offset];
    if (tag == 0xc0) {
      ++offset;
      value.kind = DecodedValue::Kind::kNil;
    } else if (tag == 0xc2 || tag == 0xc3) {
      ++offset;
      value.kind = DecodedValue::Kind::kBool;
      value.bool_value = tag == 0xc3;
    } else if ((tag & 0xe0) == 0xa0 || tag == 0xd9 || tag == 0xda ||
               tag == 0xdb) {
      auto string_value = DecodeStringAt(bytes, &offset);
      if (!string_value) {
        return std::nullopt;
      }
      value.kind = DecodedValue::Kind::kString;
      value.string_value = *string_value;
    } else if ((tag & 0xf0) == 0x90 || tag == 0xdc || tag == 0xdd) {
      auto array_value = DecodeUnsignedArrayAt(bytes, &offset);
      if (!array_value) {
        return std::nullopt;
      }
      value.kind = DecodedValue::Kind::kUnsignedArray;
      value.unsigned_array_value = std::move(*array_value);
    } else {
      auto unsigned_value = DecodeUnsignedAt(bytes, &offset);
      if (!unsigned_value) {
        return std::nullopt;
      }
      value.kind = DecodedValue::Kind::kUnsigned;
      value.unsigned_value = *unsigned_value;
    }
    out.emplace(std::move(*key), std::move(value));
  }
  if (offset != bytes.size()) {
    return std::nullopt;
  }
  return out;
}

std::optional<std::vector<DecodedBlockAllocationRecord>>
DecodeBlockAllocationRecords(const Bytes& bytes) {
  std::size_t offset = 0;
  const auto maybe_len = DecodeArrayHeader(bytes, &offset);
  if (!maybe_len) {
    return std::nullopt;
  }

  std::vector<DecodedBlockAllocationRecord> out;
  out.reserve(*maybe_len);
  for (std::size_t i = 0; i < *maybe_len; ++i) {
    const auto maybe_fields = DecodeMapHeader(bytes, &offset);
    if (!maybe_fields) {
      return std::nullopt;
    }
    DecodedBlockAllocationRecord record;
    bool saw_req_id = false;
    bool saw_block_ids = false;
    bool saw_token_ids = false;
    for (std::size_t field = 0; field < *maybe_fields; ++field) {
      auto key = DecodeStringAt(bytes, &offset);
      if (!key) {
        return std::nullopt;
      }
      if (*key == "req_id") {
        auto req_id = DecodeStringAt(bytes, &offset);
        if (!req_id) {
          return std::nullopt;
        }
        record.req_id = std::move(*req_id);
        saw_req_id = true;
      } else if (*key == "new_block_ids") {
        auto block_ids = DecodeUnsignedArrayAt(bytes, &offset);
        if (!block_ids) {
          return std::nullopt;
        }
        record.new_block_ids = std::move(*block_ids);
        saw_block_ids = true;
      } else if (*key == "new_token_ids") {
        auto token_ids = DecodeUnsignedArrayAt(bytes, &offset);
        if (!token_ids) {
          return std::nullopt;
        }
        record.new_token_ids = std::move(*token_ids);
        saw_token_ids = true;
      } else {
        return std::nullopt;
      }
    }
    if (!saw_req_id || !saw_block_ids || !saw_token_ids) {
      return std::nullopt;
    }
    out.push_back(std::move(record));
  }
  if (offset != bytes.size()) {
    return std::nullopt;
  }
  return out;
}

std::optional<std::vector<DecodedCbMatchResult>> DecodeCbMatchResults(
    const Bytes& bytes) {
  std::size_t offset = 0;
  const auto maybe_len = DecodeArrayHeader(bytes, &offset);
  if (!maybe_len) {
    return std::nullopt;
  }

  std::vector<DecodedCbMatchResult> out;
  out.reserve(*maybe_len);
  for (std::size_t i = 0; i < *maybe_len; ++i) {
    const auto maybe_fields = DecodeMapHeader(bytes, &offset);
    if (!maybe_fields) {
      return std::nullopt;
    }
    DecodedCbMatchResult result;
    bool saw_old_st = false;
    bool saw_old_ed = false;
    bool saw_cur_st = false;
    bool saw_cur_ed = false;
    bool saw_hash = false;
    for (std::size_t field = 0; field < *maybe_fields; ++field) {
      auto key = DecodeStringAt(bytes, &offset);
      if (!key) {
        return std::nullopt;
      }
      if (*key == "old_st") {
        auto value = DecodeUnsignedAt(bytes, &offset);
        if (!value) {
          return std::nullopt;
        }
        result.old_st = *value;
        saw_old_st = true;
      } else if (*key == "old_ed") {
        auto value = DecodeUnsignedAt(bytes, &offset);
        if (!value) {
          return std::nullopt;
        }
        result.old_ed = *value;
        saw_old_ed = true;
      } else if (*key == "cur_st") {
        auto value = DecodeUnsignedAt(bytes, &offset);
        if (!value) {
          return std::nullopt;
        }
        result.cur_st = *value;
        saw_cur_st = true;
      } else if (*key == "cur_ed") {
        auto value = DecodeUnsignedAt(bytes, &offset);
        if (!value) {
          return std::nullopt;
        }
        result.cur_ed = *value;
        saw_cur_ed = true;
      } else if (*key == "hash") {
        auto hash = DecodeBinaryAt(bytes, &offset);
        if (!hash) {
          return std::nullopt;
        }
        result.hash = std::move(*hash);
        saw_hash = true;
      } else {
        return std::nullopt;
      }
    }
    if (!saw_old_st || !saw_old_ed || !saw_cur_st || !saw_cur_ed || !saw_hash ||
        result.old_st > result.old_ed || result.cur_st > result.cur_ed) {
      return std::nullopt;
    }
    out.push_back(std::move(result));
  }
  if (offset != bytes.size()) {
    return std::nullopt;
  }
  return out;
}

std::optional<std::vector<DecodedCudaIpcWrapper>> DecodeCudaIpcWrapperArray(
    const Bytes& bytes) {
  std::size_t offset = 0;
  const auto maybe_len = DecodeArrayHeader(bytes, &offset);
  if (!maybe_len) {
    return std::nullopt;
  }

  std::vector<DecodedCudaIpcWrapper> out;
  out.reserve(*maybe_len);
  for (std::size_t i = 0; i < *maybe_len; ++i) {
    auto ext = DecodeExtAt(bytes, &offset);
    if (!ext || ext->first != 1) {
      return std::nullopt;
    }

    std::size_t payload_offset = 0;
    const auto maybe_fields = DecodeMapHeader(ext->second, &payload_offset);
    if (!maybe_fields) {
      return std::nullopt;
    }

    DecodedCudaIpcWrapper wrapper;
    bool saw_magic = false;
    bool saw_version = false;
    bool saw_kind = false;
    bool saw_dtype = false;
    bool saw_shape = false;
    bool saw_stride = false;
    bool saw_storage_offset = false;
    bool saw_device_uuid = false;
    for (std::size_t field = 0; field < *maybe_fields; ++field) {
      auto key = DecodeStringAt(ext->second, &payload_offset);
      if (!key) {
        return std::nullopt;
      }
      if (*key == "magic") {
        auto value = DecodeStringAt(ext->second, &payload_offset);
        if (!value || *value != "lmcache.cuda_ipc_wrapper") {
          return std::nullopt;
        }
        saw_magic = true;
      } else if (*key == "version") {
        auto value = DecodeUnsignedAt(ext->second, &payload_offset);
        if (!value || *value != 1) {
          return std::nullopt;
        }
        saw_version = true;
      } else if (*key == "kind") {
        auto value = DecodeStringAt(ext->second, &payload_offset);
        if (!value || (*value != "cuda" && *value != "raw")) {
          return std::nullopt;
        }
        wrapper.kind = std::move(*value);
        saw_kind = true;
      } else if (*key == "dtype") {
        auto value = DecodeStringAt(ext->second, &payload_offset);
        if (!value) {
          return std::nullopt;
        }
        wrapper.dtype = std::move(*value);
        saw_dtype = true;
      } else if (*key == "shape") {
        auto value = DecodeUnsignedArrayAt(ext->second, &payload_offset);
        if (!value) {
          return std::nullopt;
        }
        wrapper.shape = std::move(*value);
        saw_shape = true;
      } else if (*key == "stride") {
        auto value = DecodeUnsignedArrayAt(ext->second, &payload_offset);
        if (!value) {
          return std::nullopt;
        }
        wrapper.stride = std::move(*value);
        saw_stride = true;
      } else if (*key == "storage_offset") {
        auto value = DecodeUnsignedAt(ext->second, &payload_offset);
        if (!value) {
          return std::nullopt;
        }
        wrapper.storage_offset = *value;
        saw_storage_offset = true;
      } else if (*key == "device_uuid") {
        auto value = DecodeStringAt(ext->second, &payload_offset);
        if (!value) {
          return std::nullopt;
        }
        wrapper.device_uuid = std::move(*value);
        saw_device_uuid = true;
      } else if (*key == "handle") {
        if (payload_offset < ext->second.size() &&
            ext->second[payload_offset] == 0xc0) {
          ++payload_offset;
        } else if (!DecodeCudaHandleAt(ext->second, &payload_offset,
                                       &wrapper)) {
          return std::nullopt;
        }
      } else if (*key == "ipc_handle_reserved") {
        auto value = DecodeBinaryAt(ext->second, &payload_offset);
        if (!value) {
          return std::nullopt;
        }
        wrapper.ipc_handle_bytes = value->size();
        wrapper.ipc_handle = std::move(*value);
      } else if (*key == "nbytes") {
        auto value = DecodeUnsignedAt(ext->second, &payload_offset);
        if (!value) {
          return std::nullopt;
        }
        wrapper.raw_nbytes = *value;
      } else if (!SkipValueAt(ext->second, &payload_offset)) {
        return std::nullopt;
      }
    }
    if (payload_offset != ext->second.size() || !saw_magic || !saw_version ||
        !saw_kind || !saw_dtype || !saw_shape || !saw_stride ||
        !saw_storage_offset || !saw_device_uuid ||
        wrapper.shape.size() != wrapper.stride.size()) {
      return std::nullopt;
    }
    out.push_back(std::move(wrapper));
  }
  if (offset != bytes.size()) {
    return std::nullopt;
  }
  return out;
}

Bytes EncodeNil() { return Bytes{0xc0}; }

Bytes EncodeBool(bool value) {
  return Bytes{static_cast<std::uint8_t>(value ? 0xc3 : 0xc2)};
}

Bytes EncodeEmptyArray() { return Bytes{0x90}; }

Bytes EncodeUnsigned(std::uint64_t value) {
  Bytes out;
  if (value <= 0x7f) {
    out.push_back(static_cast<std::uint8_t>(value));
  } else if (value <= std::numeric_limits<std::uint8_t>::max()) {
    out.push_back(0xcc);
    out.push_back(static_cast<std::uint8_t>(value));
  } else if (value <= std::numeric_limits<std::uint16_t>::max()) {
    out.push_back(0xcd);
    AppendU16(out, static_cast<std::uint16_t>(value));
  } else if (value <= std::numeric_limits<std::uint32_t>::max()) {
    out.push_back(0xce);
    AppendU32(out, static_cast<std::uint32_t>(value));
  } else {
    out.push_back(0xcf);
    AppendU64(out, value);
  }
  return out;
}

Bytes EncodeString(const std::string& value) {
  Bytes out;
  const auto len = value.size();
  if (len <= 31) {
    out.push_back(static_cast<std::uint8_t>(0xa0 | len));
  } else if (len <= std::numeric_limits<std::uint8_t>::max()) {
    out.push_back(0xd9);
    out.push_back(static_cast<std::uint8_t>(len));
  } else if (len <= std::numeric_limits<std::uint16_t>::max()) {
    out.push_back(0xda);
    AppendU16(out, static_cast<std::uint16_t>(len));
  } else {
    out.push_back(0xdb);
    AppendU32(out, static_cast<std::uint32_t>(len));
  }
  out.insert(out.end(), value.begin(), value.end());
  return out;
}

Bytes EncodeBytes(const Bytes& value) {
  Bytes out;
  const auto len = value.size();
  if (len <= std::numeric_limits<std::uint8_t>::max()) {
    out.push_back(0xc4);
    out.push_back(static_cast<std::uint8_t>(len));
  } else if (len <= std::numeric_limits<std::uint16_t>::max()) {
    out.push_back(0xc5);
    AppendU16(out, static_cast<std::uint16_t>(len));
  } else {
    out.push_back(0xc6);
    AppendU32(out, static_cast<std::uint32_t>(len));
  }
  out.insert(out.end(), value.begin(), value.end());
  return out;
}

Bytes EncodeBytesBoolTuple(const Bytes& bytes, bool value) {
  Bytes out{0x92};
  Bytes encoded_bytes = EncodeBytes(bytes);
  Bytes encoded_bool = EncodeBool(value);
  out.insert(out.end(), encoded_bytes.begin(), encoded_bytes.end());
  out.insert(out.end(), encoded_bool.begin(), encoded_bool.end());
  return out;
}

}  // namespace lmcache::mp::msgpack
