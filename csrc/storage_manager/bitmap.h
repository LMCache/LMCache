#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>

namespace lmcache {

namespace storage_manager {

/**
 * @brief A simple bitmap implementation for tracking the state of the L2
 * storage operation results.
 *
 * This bitmap is used to track the state of the L2 storage operation results.
 * Each bit in the bitmap represents the success or failure of a key.
 */
class Bitmap {
 public:
  /**
   * @brief Construct a new Bitmap with the specified size.
   *
   * @param size The number of bits in the bitmap.
   */
  explicit Bitmap(size_t size);

  /**
   * @brief set the bit at the specified index to 1.
   */
  void set(size_t index);

  /**
   * @brief clear the bit at the specified index to 0.
   */
  void clear(size_t index);

  /**
   * @brief test the bit at the specified index.
   *
   * @return true if the bit is set to 1, false otherwise.
   */
  bool test(size_t index) const;

  /**
   * @brief count the number of bits set to 1 in the bitmap.
   *
   * @return the number of bits set to 1.
   */
  size_t popcount() const;

  /**
   * @brief count the number of leading zeros in the bitmap.
   *
   * @return the number of leading zeros.
   */
  size_t clz() const;

  /**
   * @brief count the number of leading ones in the bitmap.
   *
   * @return the number of leading ones.
   */
  size_t clo() const;

  /**
   * @brief bitwise AND operation between two bitmaps.
   *
   * @return a new Bitmap that is the result of the bitwise AND operation.
   *
   * @note If this and other have different sizes, the result will be truncated
   * to the smaller size.
   */
  Bitmap operator&(const Bitmap& other) const;

  /**
   * @brief convert the bitmap to a string representation.
   *
   * @return a string representation of the bitmap, where '1' represents a set
   * bit and '0' represents a clear bit.
   */
  std::string to_string() const;

  /**
   * @brief Destructor.
   */
  ~Bitmap();

 private:
  size_t size_;
  std::vector<uint8_t> data_;

  // Helper functions

  /**
   * @brief flip the bits in the bitmap (bitwise NOT).
   *
   * @return a new Bitmap that is the result of the bitwise NOT operation.
   */
  Bitmap operator~() const;
};

}  // namespace storage_manager

}  // namespace lmcache
