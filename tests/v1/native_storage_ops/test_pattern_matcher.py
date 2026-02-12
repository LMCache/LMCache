# SPDX-License-Identifier: Apache-2.0

# Standard
from concurrent.futures import ThreadPoolExecutor
import threading

# Third Party
import pytest
import torch

pytest.importorskip(
    "lmcache.native_storage_ops",
    reason="native_storage_ops extension not built",
)

# First Party
from lmcache.native_storage_ops import ParallelPatternMatcher


def assert_torch():
    """Test that torch is available and can be imported."""
    assert torch.__version__ is not None, "Torch should be available"


class TestParallelPatternMatcherBasic:
    """Test basic functionality of ParallelPatternMatcher."""

    def test_example_from_spec(self):
        """Test the exact example from the specification."""
        pattern = [1, 2, 3]
        data = [1, 1, 1, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [3, 8], f"Expected [3, 8] but got {result}"

    def test_no_matches(self):
        """Test when pattern is not found in data."""
        pattern = [5, 6, 7]
        data = [1, 2, 3, 4, 1, 2, 3, 4]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == []

    def test_single_match(self):
        """Test when pattern appears once."""
        pattern = [7, 8, 9]
        data = [1, 2, 3, 7, 8, 9, 10, 11]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [3]

    def test_pattern_at_start(self):
        """Test when pattern is at the beginning of data."""
        pattern = [1, 2, 3]
        data = [1, 2, 3, 4, 5, 6]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [0]

    def test_pattern_at_end(self):
        """Test when pattern is at the end of data."""
        pattern = [4, 5, 6]
        data = [1, 2, 3, 4, 5, 6]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [3]

    def test_multiple_matches(self):
        """Test when pattern appears multiple times."""
        pattern = [1, 2]
        data = [1, 2, 3, 1, 2, 4, 1, 2, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [0, 3, 6]

    def test_overlapping_patterns(self):
        """Test when patterns overlap."""
        pattern = [1, 1]
        data = [1, 1, 1, 1, 2]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        # Overlapping patterns at positions 0, 1, 2
        assert result == [0, 1, 2]

    def test_single_element_pattern(self):
        """Test with a single-element pattern."""
        pattern = [5]
        data = [1, 2, 5, 3, 5, 4, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [2, 4, 6]

    def test_pattern_equals_data(self):
        """Test when pattern is exactly the same as data."""
        pattern = [1, 2, 3, 4, 5]
        data = [1, 2, 3, 4, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [0]

    def test_pattern_longer_than_data(self):
        """Test when pattern is longer than data."""
        pattern = [1, 2, 3, 4, 5]
        data = [1, 2, 3]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == []

    def test_empty_data(self):
        """Test with empty data."""
        pattern = [1, 2, 3]
        data = []

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == []

    def test_large_pattern(self):
        """Test with a large pattern."""
        pattern = list(range(10))
        data = [0] + list(range(10)) + [99] + list(range(10)) + [88]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [1, 12]


class TestParallelPatternMatcherConsistency:
    """Test consistency of pattern matching."""

    def test_repeated_matches_consistent(self):
        """Test that repeated calls produce consistent results."""
        pattern = [1, 2, 3]
        data = [1, 1, 1, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3]

        matcher = ParallelPatternMatcher(pattern)

        # Call match multiple times
        results = [matcher.match(data) for _ in range(5)]

        # All results should be the same
        for result in results:
            assert result == [3, 8]

    def test_multiple_matchers_same_result(self):
        """Test that multiple matcher instances produce the same result."""
        pattern = [5, 6, 7]
        data = list(range(100)) + [5, 6, 7] + list(range(50)) + [5, 6, 7]

        results = []
        for _ in range(5):
            matcher = ParallelPatternMatcher(pattern)
            result = matcher.match(data)
            results.append(result)

        # All results should be the same
        for i in range(1, len(results)):
            assert results[i] == results[0], (
                f"Results differ: {results[0]} vs {results[i]}"
            )


class TestParallelPatternMatcherEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_numbers(self):
        """Test with negative numbers in pattern and data."""
        pattern = [-1, -2, -3]
        data = [1, 2, -1, -2, -3, 4, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [2]

    def test_mixed_positive_negative(self):
        """Test with mixed positive and negative numbers."""
        pattern = [1, -1, 2]
        data = [0, 1, -1, 2, 3, 1, -1, 2, 4]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [1, 5]

    def test_zeros_in_pattern(self):
        """Test with zeros in the pattern."""
        pattern = [0, 0, 0]
        data = [1, 2, 0, 0, 0, 3, 0, 0, 0]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [2, 6]

    def test_large_numbers(self):
        """Test with large numbers."""
        pattern = [1000000, 2000000, 3000000]
        data = [1, 1000000, 2000000, 3000000, 2, 1000000, 2000000, 3000000]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [1, 5]

    def test_repeating_pattern_in_repeating_data(self):
        """Test repeating pattern in repeating data."""
        pattern = [1, 2]
        data = [1, 2, 1, 2, 1, 2, 1, 2]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [0, 2, 4, 6]

    def test_pattern_at_every_position(self):
        """Test when pattern could match at multiple overlapping positions."""
        pattern = [1, 1, 1]
        data = [1, 1, 1, 1, 1]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        # Pattern matches at positions 0, 1, 2
        assert result == [0, 1, 2]


class TestParallelPatternMatcherThreadSafety:
    """Test thread safety when using the same matcher from multiple threads."""

    def test_concurrent_matches(self):
        """Test that multiple threads can call match() concurrently."""
        pattern = [1, 2, 3]
        data = [1, 1, 1, 1, 2, 3, 3, 3, 1, 2, 3, 3, 3]

        matcher = ParallelPatternMatcher(pattern)

        def do_match():
            result = matcher.match(data)
            assert result == [3, 8]

        num_threads = 10
        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=do_match)
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    def test_concurrent_different_data(self):
        """Test concurrent matches with different data."""
        pattern = [5, 6, 7]
        matcher = ParallelPatternMatcher(pattern)

        data_sets = [
            ([5, 6, 7, 1, 2, 3], [0]),
            ([1, 2, 5, 6, 7, 8], [2]),
            ([1, 2, 3, 4, 5, 6], []),
            ([5, 6, 7, 5, 6, 7], [0, 3]),
        ]

        results = [None] * len(data_sets)

        def do_match(idx, data, expected):
            result = matcher.match(data)
            assert result == expected, f"Expected {expected} but got {result}"
            results[idx] = result

        threads = []
        for idx, (data, expected) in enumerate(data_sets):
            t = threading.Thread(target=do_match, args=(idx, data, expected))
            threads.append(t)

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Verify all results
        for idx, (_, expected) in enumerate(data_sets):
            assert results[idx] == expected


class TestParallelPatternMatcherPerformance:
    """Test performance characteristics."""

    def test_large_data(self):
        """Test with large data."""
        pattern = [42, 43, 42]
        data = list(range(10000)) + [42, 43, 42] + list(range(5000))

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == [10000]

    def test_many_matches_in_large_data(self):
        """Test with many pattern occurrences in large data."""
        pattern = [1, 2, 3]
        # Create data with pattern appearing every 10 elements
        data = []
        for i in range(1000):
            if i % 10 == 0:
                data.extend([1, 2, 3])
            else:
                data.extend([4, 5, 6])

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        # Verify we found all occurrences
        for idx in result:
            assert data[idx : idx + 3] == [1, 2, 3], f"Invalid match at position {idx}"

        # Should have ~100 matches
        assert len(result) > 90

    def test_stress_test_parallel_execution(self):
        """Stress test with multiple matchers running in parallel."""
        pattern = [7, 8, 9]
        data = list(range(1000)) + [7, 8, 9] + list(range(500))

        def worker():
            matcher = ParallelPatternMatcher(pattern)
            result = matcher.match(data)
            assert 1000 in result

        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(worker) for _ in range(100)]
            for f in futures:
                f.result()  # Will raise if any thread failed


class TestParallelPatternMatcherResultSorted:
    """Test that results are always sorted."""

    def test_results_are_sorted(self):
        """Test that matches are returned in sorted order."""
        pattern = [1, 2]
        # Create data where pattern appears in various positions
        data = [0, 1, 2, 0, 1, 2, 0, 0, 1, 2, 0]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        # Verify sorted
        assert result == sorted(result)

        # Verify correctness
        assert result == [1, 4, 8]

    def test_results_sorted_multiple_matches(self):
        """Test that results are sorted with multiple matches."""
        pattern = [5]
        data = [5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5]

        matcher = ParallelPatternMatcher(pattern)
        result = matcher.match(data)

        assert result == sorted(result)
        assert result == [0, 2, 4, 6, 8, 10]
