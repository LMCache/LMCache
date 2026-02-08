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
    : size_(size), data_(size == 0 ? 0 : (size - 1) / kBitsPerByte + 1, 0) {}

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

  // process full bytes
  size_t count = 0;
  const size_t num_full_bytes = size_ / kBitsPerByte;
  for (size_t i = 0; i < num_full_bytes; ++i) {
    count += static_cast<size_t>(
        __builtin_popcount(static_cast<unsigned>(data_[i])));
  }

  // process remaining bits in the last byte
  const unsigned remaining_bits = size_ % kBitsPerByte;
  if (remaining_bits > 0) {
    uint8_t last_byte = data_.back();
    uint8_t mask = static_cast<uint8_t>((1u << remaining_bits) - 1);
    last_byte &= mask;
    count += static_cast<size_t>(
        __builtin_popcount(static_cast<unsigned>(last_byte)));
  }
  return count;
}

size_t Bitmap::clz() const {
  if (size_ == 0) return 0;

  const size_t num_full_bytes = size_ / kBitsPerByte;
  size_t count = 0;

  for (size_t i = 0; i < num_full_bytes; ++i) {
    uint8_t b = data_[i];
    if (b == 0) {
      count += kBitsPerByte;
    } else {
      count += static_cast<size_t>(__builtin_ctz(static_cast<unsigned>(b)));
      return count;
    }
  }

  const unsigned remaining_bits = size_ % kBitsPerByte;
  if (remaining_bits > 0) {
    uint8_t last_byte = data_.back();
    uint8_t mask = static_cast<uint8_t>((1u << remaining_bits) - 1);
    last_byte &= mask;
    if (last_byte == 0) {
      count += remaining_bits;
    } else {
      count +=
          static_cast<size_t>(__builtin_ctz(static_cast<unsigned>(last_byte)));
    }
  }

  return count;
}

size_t Bitmap::clo() const {
  const Bitmap inverted{~(*this)};
  return inverted.clz();
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
  if (size_ == 0) return "";

  std::string result(size_, '0');
  for (size_t i = 0; i < data_.size(); ++i) {
    uint8_t byte = data_[i];
    for (unsigned j = 0; j < kBitsPerByte; ++j) {
      size_t bit_index = i * kBitsPerByte + j;
      if (bit_index >= size_) {
        break;
      }
      if ((byte >> j) & 1u) {
        result[bit_index] = '1';
      }
    }
  }

  return result;
}

Bitmap Bitmap::operator~() const {
  Bitmap result(size_);
  for (size_t i = 0; i < data_.size(); ++i) {
    result.data_[i] = ~data_[i];
  }

  // Clear bits that are out of range in the last byte
  const unsigned remaining_bits = size_ % kBitsPerByte;
  if (remaining_bits > 0) {
    uint8_t mask = static_cast<uint8_t>((1u << remaining_bits) - 1);
    result.data_.back() &= mask;
  }
  return result;
}

Bitmap::~Bitmap() = default;

}  // namespace storage_manager
}  // namespace lmcache
