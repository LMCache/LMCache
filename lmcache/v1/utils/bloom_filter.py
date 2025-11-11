# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Union
import array
import hashlib

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


class BloomFilter:
    """
    A simple Bloom Filter implementation for memory-efficient set membership testing.

    Args:
        expected_elements: Expected number of elements to store
        false_positive_rate: Desired false positive rate (default 0.01 = 1%)
    """

    def __init__(
        self, expected_elements: int = 1000000, false_positive_rate: float = 0.01
    ):
        # Calculate optimal bit array size and number of hash functions
        self.size = self._optimal_size(expected_elements, false_positive_rate)
        self.hash_count = self._optimal_hash_count(self.size, expected_elements)
        # Use array of 32-bit integers for better performance
        array_size = (self.size + 31) // 32
        self.bit_array = array.array("I", [0] * array_size)
        self.item_count = 0  # Track number of items added
        self.expected_elements = expected_elements
        self.false_positive_rate = false_positive_rate

        logger.info(
            "BloomFilter initialized with size=%d bits "
            "(%.2f MB), "
            "hash_count=%d, "
            "expected_elements=%d, "
            "false_positive_rate=%f",
            self.size,
            self.size / 8 / 1024 / 1024,
            self.hash_count,
            expected_elements,
            false_positive_rate,
        )

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """Calculate optimal bit array size."""
        # Standard
        import math

        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """Calculate optimal number of hash functions."""
        # Standard
        import math

        k = (m / n) * math.log(2)
        return max(1, int(k))

    def _hashes(self, item: Union[str, int]) -> list[int]:
        """Generate multiple hash values for an item."""
        result = []
        if isinstance(item, int):
            for i in range(self.hash_count):
                h = hashlib.sha256(
                    item.to_bytes((item.bit_length() + 7) // 8, "big", signed=False)
                    + i.to_bytes(4, "big")
                ).digest()
                result.append(int.from_bytes(h[:4], "big") % self.size)
        else:
            for i in range(self.hash_count):
                h = hashlib.sha256(f"{item}_{i}".encode()).digest()
                result.append(int.from_bytes(h[:4], "big") % self.size)
        return result

    def add(self, item: Union[str, int]) -> None:
        """Add an item to the Bloom Filter."""
        for pos in self._hashes(item):
            idx = pos >> 5
            bit = pos & 31
            self.bit_array[idx] |= 1 << bit
        self.item_count += 1

    def add_with_hashes(self, hash_positions: list[int]) -> None:
        """Add an item using pre-computed hash positions."""
        for pos in hash_positions:
            idx = pos >> 5
            bit = pos & 31
            self.bit_array[idx] |= 1 << bit
        self.item_count += 1

    def contains(self, item: Union[str, int]) -> bool:
        """Check if an item might be in the set (may have false positives)."""
        for pos in self._hashes(item):
            idx = pos >> 5
            bit = pos & 31
            if not (self.bit_array[idx] & (1 << bit)):
                return False
        return True

    def contains_with_hashes(self, hash_positions: list[int]) -> bool:
        """Check if an item might be in the set using pre-computed hash positions."""
        for pos in hash_positions:
            idx = pos >> 5
            bit = pos & 31
            if not (self.bit_array[idx] & (1 << bit)):
                return False
        return True

    def clear(self) -> None:
        """Clear all items from the Bloom Filter."""
        array_size = (self.size + 31) // 32
        self.bit_array = array.array("I", [0] * array_size)
        self.item_count = 0

    def get_memory_usage_bytes(self) -> int:
        """Get the memory usage of the Bloom Filter in bytes."""
        # Each boolean in Python list takes ~28 bytes (object overhead)
        # But bit_array is a list of booleans, actual memory is size / 8
        return self.size // 8

    def get_statistics(self) -> dict:
        """Get Bloom Filter statistics."""
        bits_set = sum(bin(val).count("1") for val in self.bit_array)
        fill_rate = bits_set / self.size if self.size > 0 else 0.0
        size_bytes = self.get_memory_usage_bytes()
        return {
            "size_mb": size_bytes / 1024 / 1024,
            "hash_count": self.hash_count,
            "item_count": self.item_count,
            "bits_set": bits_set,
            "fill_rate": fill_rate,
            "expected_elements": self.expected_elements,
            "false_positive_rate": self.false_positive_rate,
        }
