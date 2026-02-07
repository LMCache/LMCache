// SPDX-License-Identifier: Apache-2.0

#include "bitmap.h"

namespace lmcache {
namespace storage_manager {

namespace {

constexpr unsigned kBitsPerByte = 8;

inline size_t byte_index(size_t bit_index) { return bit_index / kBitsPerByte; }
inline unsigned bit_offset(size_t bit_index) {
  return static_cast<unsigned>(bit_index % kBitsPerByte);
}

}  // namespace

Bitmap::Bitmap(size_t size)
    : size_(size), data_((size + kBitsPerByte - 1) / kBitsPerByte, 0) {}

void Bitmap::set(size_t index) {
  if (index >= size_) return;
  data_[byte_index(index)] |= static_cast<uint8_t>(1u << bit_offset(index));
}

void Bitmap::clear(size_t index) {
  if (index >= size_) return;
  data_[byte_index(index)] &= static_cast<uint8_t>(~(1u << bit_offset(index)));
}

bool Bitmap::test(size_t index) const {
  if (index >= size_) return false;
  return (data_[byte_index(index)] >> bit_offset(index)) & 1u;
}

size_t Bitmap::popcount() const {
  if (data_.empty()) return 0;
  size_t count = 0;
  const size_t num_bytes = data_.size();
  for (size_t i = 0; i < num_bytes; ++i) {
    uint8_t b = data_[i];
    if (i == num_bytes - 1 && size_ % kBitsPerByte != 0) {
      b &= static_cast<uint8_t>((1u << (size_ % kBitsPerByte)) - 1);
    }
    count += static_cast<size_t>(__builtin_popcount(static_cast<unsigned>(b)));
  }
  return count;
}

size_t Bitmap::clz() const {
  const size_t num_bytes = data_.size();
  const unsigned last_byte_bits =
      (num_bytes > 0 && size_ % kBitsPerByte != 0)
          ? static_cast<unsigned>(size_ % kBitsPerByte)
          : kBitsPerByte;
  size_t count = 0;
  for (size_t i = 0; i < num_bytes; ++i) {
    const bool is_last_byte = (i == num_bytes - 1);
    const unsigned bits_in_byte = is_last_byte ? last_byte_bits : kBitsPerByte;
    uint8_t b = data_[i];
    if (is_last_byte && bits_in_byte < kBitsPerByte) {
      b &= static_cast<uint8_t>((1u << bits_in_byte) - 1);
    }
    if (b == 0) {
      count += bits_in_byte;
    } else {
      count += static_cast<size_t>(__builtin_ctz(static_cast<unsigned>(b)));
      return count;
    }
  }
  return count;
}

size_t Bitmap::clo() const {
  const size_t num_bytes = data_.size();
  const unsigned last_byte_bits =
      (num_bytes > 0 && size_ % kBitsPerByte != 0)
          ? static_cast<unsigned>(size_ % kBitsPerByte)
          : kBitsPerByte;
  size_t count = 0;
  for (size_t i = 0; i < num_bytes; ++i) {
    const bool is_last_byte = (i == num_bytes - 1);
    const unsigned bits_in_byte = is_last_byte ? last_byte_bits : kBitsPerByte;
    uint8_t b = data_[i];
    if (is_last_byte && bits_in_byte < kBitsPerByte) {
      b &= static_cast<uint8_t>((1u << bits_in_byte) - 1);
    }
    const uint8_t full_byte =
        (bits_in_byte == kBitsPerByte)
            ? 0xFF
            : static_cast<uint8_t>((1u << bits_in_byte) - 1);
    if (b == full_byte) {
      count += bits_in_byte;
    } else {
      const unsigned mask = (1u << bits_in_byte) - 1;
      count += static_cast<size_t>(
          __builtin_ctz(static_cast<unsigned>((~b) & mask)));
      return count;
    }
  }
  return count;
}

Bitmap Bitmap::operator&(const Bitmap& other) const {
  const size_t result_size = (size_ <= other.size_) ? size_ : other.size_;
  Bitmap result(result_size);
  for (size_t i = 0; i < result.data_.size(); ++i) {
    result.data_[i] = data_[i] & other.data_[i];
  }
  return result;
}

std::string Bitmap::to_string() const {
  std::string result(size_, '0');
  for (size_t i = 0; i < size_; ++i) {
    if (test(i)) {
      result[i] = '1';
    }
  }
  return result;
}

Bitmap::~Bitmap() = default;

}  // namespace storage_manager
}  // namespace lmcache
