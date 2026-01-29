# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for error classes in distributed storage manager.

These tests verify the behavior of:

1. L1MemoryManagerError - Simple enum for memory manager errors
2. L1ObjectManagerError - Bitwise-combinable enum for object manager errors
   - has_error() - Check if a specific error is present
   - mix_error() - Combine multiple errors using bitwise OR
3. strerror() - Convert error codes to human-readable strings
"""

# Third Party

# First Party
from lmcache.v1.multiprocess.distributed.error import (
    L1MemoryManagerError,
    L1ObjectManagerError,
    strerror,
)

# =============================================================================
# L1MemoryManagerError Tests
# =============================================================================


class TestL1MemoryManagerError:
    """Tests for L1MemoryManagerError enum."""

    def test_success_value(self):
        """SUCCESS should be a valid enum value."""
        assert L1MemoryManagerError.SUCCESS is not None
        assert isinstance(L1MemoryManagerError.SUCCESS, L1MemoryManagerError)

    def test_out_of_memory_value(self):
        """OUT_OF_MEMORY should be a valid enum value."""
        assert L1MemoryManagerError.OUT_OF_MEMORY is not None
        assert isinstance(L1MemoryManagerError.OUT_OF_MEMORY, L1MemoryManagerError)

    def test_success_and_out_of_memory_are_different(self):
        """SUCCESS and OUT_OF_MEMORY should have different values."""
        assert L1MemoryManagerError.SUCCESS != L1MemoryManagerError.OUT_OF_MEMORY

    def test_enum_members_count(self):
        """L1MemoryManagerError should have exactly 2 members."""
        assert len(L1MemoryManagerError) == 2


# =============================================================================
# L1ObjectManagerError Tests
# =============================================================================


class TestL1ObjectManagerError:
    """Tests for L1ObjectManagerError IntFlag."""

    def test_success_is_zero(self):
        """SUCCESS should have value 0x00."""
        assert L1ObjectManagerError.SUCCESS.value == 0x00

    def test_error_values_are_powers_of_two(self):
        """All error values (except SUCCESS) should be powers of two."""
        expected_values = {
            L1ObjectManagerError.SUCCESS: 0x00,
            L1ObjectManagerError.KEYS_NOT_FOUND: 0x01,
            L1ObjectManagerError.KEYS_ALREADY_EXIST: 0x02,
            L1ObjectManagerError.KEYS_NOT_RESERVED: 0x04,
            L1ObjectManagerError.KEYS_NOT_COMMITTED: 0x08,
            L1ObjectManagerError.KEYS_ALREADY_LOCKED: 0x10,
        }
        for error, expected_value in expected_values.items():
            assert error.value == expected_value, f"{error} has unexpected value"

    def test_all_members_exist(self):
        """All expected L1ObjectManagerError members should exist."""
        # IntFlag with SUCCESS=0 may not count it in len(), so check members directly
        assert hasattr(L1ObjectManagerError, "SUCCESS")
        assert hasattr(L1ObjectManagerError, "KEYS_NOT_FOUND")
        assert hasattr(L1ObjectManagerError, "KEYS_ALREADY_EXIST")
        assert hasattr(L1ObjectManagerError, "KEYS_NOT_RESERVED")
        assert hasattr(L1ObjectManagerError, "KEYS_NOT_COMMITTED")
        assert hasattr(L1ObjectManagerError, "KEYS_ALREADY_LOCKED")

    def test_is_intflag(self):
        """L1ObjectManagerError should be an IntFlag for bitwise operations."""
        # Standard
        import enum

        assert issubclass(L1ObjectManagerError, enum.IntFlag)


class TestL1ObjectManagerErrorHasError:
    """Tests for L1ObjectManagerError.has_error() method."""

    def test_success_has_no_errors(self):
        """SUCCESS should not have any error flags set."""
        success = L1ObjectManagerError.SUCCESS
        assert not success.has_error(L1ObjectManagerError.KEYS_NOT_FOUND)
        assert not success.has_error(L1ObjectManagerError.KEYS_ALREADY_EXIST)
        assert not success.has_error(L1ObjectManagerError.KEYS_NOT_RESERVED)
        assert not success.has_error(L1ObjectManagerError.KEYS_NOT_COMMITTED)
        assert not success.has_error(L1ObjectManagerError.KEYS_ALREADY_LOCKED)

    def test_single_error_has_itself(self):
        """Each single error should have itself."""
        errors = [
            L1ObjectManagerError.KEYS_NOT_FOUND,
            L1ObjectManagerError.KEYS_ALREADY_EXIST,
            L1ObjectManagerError.KEYS_NOT_RESERVED,
            L1ObjectManagerError.KEYS_NOT_COMMITTED,
            L1ObjectManagerError.KEYS_ALREADY_LOCKED,
        ]
        for error in errors:
            assert error.has_error(error), f"{error} should have itself"

    def test_single_error_does_not_have_other_errors(self):
        """A single error should not have other error flags set."""
        error = L1ObjectManagerError.KEYS_NOT_FOUND
        assert not error.has_error(L1ObjectManagerError.KEYS_ALREADY_EXIST)
        assert not error.has_error(L1ObjectManagerError.KEYS_NOT_RESERVED)
        assert not error.has_error(L1ObjectManagerError.KEYS_NOT_COMMITTED)
        assert not error.has_error(L1ObjectManagerError.KEYS_ALREADY_LOCKED)

    def test_has_error_with_success_always_false(self):
        """has_error(SUCCESS) should always return False (0 & anything = 0)."""
        errors = [
            L1ObjectManagerError.SUCCESS,
            L1ObjectManagerError.KEYS_NOT_FOUND,
            L1ObjectManagerError.KEYS_ALREADY_EXIST,
        ]
        for error in errors:
            assert not error.has_error(L1ObjectManagerError.SUCCESS)


class TestL1ObjectManagerErrorMixError:
    """Tests for L1ObjectManagerError.mix_error() method."""

    def test_mix_success_with_error(self):
        """Mixing SUCCESS with an error should return that error."""
        result = L1ObjectManagerError.SUCCESS.mix_error(
            L1ObjectManagerError.KEYS_NOT_FOUND
        )
        assert result == L1ObjectManagerError.KEYS_NOT_FOUND

    def test_mix_error_with_success(self):
        """Mixing an error with SUCCESS should return the original error."""
        result = L1ObjectManagerError.KEYS_NOT_FOUND.mix_error(
            L1ObjectManagerError.SUCCESS
        )
        assert result == L1ObjectManagerError.KEYS_NOT_FOUND

    def test_mix_two_different_errors(self):
        """Mixing two different errors should combine their bits."""
        result = L1ObjectManagerError.KEYS_NOT_FOUND.mix_error(
            L1ObjectManagerError.KEYS_ALREADY_EXIST
        )
        # 0x01 | 0x02 = 0x03
        assert result.value == 0x03
        assert result.has_error(L1ObjectManagerError.KEYS_NOT_FOUND)
        assert result.has_error(L1ObjectManagerError.KEYS_ALREADY_EXIST)
        assert not result.has_error(L1ObjectManagerError.KEYS_NOT_RESERVED)

    def test_mix_multiple_errors(self):
        """Mixing multiple errors should combine all their bits."""
        result = L1ObjectManagerError.SUCCESS
        result = result.mix_error(L1ObjectManagerError.KEYS_NOT_FOUND)
        result = result.mix_error(L1ObjectManagerError.KEYS_NOT_COMMITTED)
        result = result.mix_error(L1ObjectManagerError.KEYS_ALREADY_LOCKED)

        # 0x01 | 0x08 | 0x10 = 0x19
        assert result.value == 0x19
        assert result.has_error(L1ObjectManagerError.KEYS_NOT_FOUND)
        assert result.has_error(L1ObjectManagerError.KEYS_NOT_COMMITTED)
        assert result.has_error(L1ObjectManagerError.KEYS_ALREADY_LOCKED)
        assert not result.has_error(L1ObjectManagerError.KEYS_ALREADY_EXIST)
        assert not result.has_error(L1ObjectManagerError.KEYS_NOT_RESERVED)

    def test_mix_same_error_is_idempotent(self):
        """Mixing the same error twice should not change the result."""
        error = L1ObjectManagerError.KEYS_NOT_FOUND
        result = error.mix_error(L1ObjectManagerError.KEYS_NOT_FOUND)
        assert result == error

    def test_mix_all_errors(self):
        """Mixing all errors should produce the maximum combined value."""
        result = L1ObjectManagerError.SUCCESS
        result = result.mix_error(L1ObjectManagerError.KEYS_NOT_FOUND)
        result = result.mix_error(L1ObjectManagerError.KEYS_ALREADY_EXIST)
        result = result.mix_error(L1ObjectManagerError.KEYS_NOT_RESERVED)
        result = result.mix_error(L1ObjectManagerError.KEYS_NOT_COMMITTED)
        result = result.mix_error(L1ObjectManagerError.KEYS_ALREADY_LOCKED)

        # 0x01 | 0x02 | 0x04 | 0x08 | 0x10 = 0x1F
        assert result.value == 0x1F
        assert result.has_error(L1ObjectManagerError.KEYS_NOT_FOUND)
        assert result.has_error(L1ObjectManagerError.KEYS_ALREADY_EXIST)
        assert result.has_error(L1ObjectManagerError.KEYS_NOT_RESERVED)
        assert result.has_error(L1ObjectManagerError.KEYS_NOT_COMMITTED)
        assert result.has_error(L1ObjectManagerError.KEYS_ALREADY_LOCKED)


class TestL1ObjectManagerErrorBitwiseOperator:
    """Tests for native bitwise OR operator on L1ObjectManagerError."""

    def test_bitwise_or_two_errors(self):
        """Native | operator should combine two errors."""
        result = (
            L1ObjectManagerError.KEYS_NOT_FOUND
            | L1ObjectManagerError.KEYS_ALREADY_EXIST
        )
        assert result.value == 0x03
        assert L1ObjectManagerError.KEYS_NOT_FOUND in result
        assert L1ObjectManagerError.KEYS_ALREADY_EXIST in result

    def test_bitwise_or_chain(self):
        """Chaining | operators should combine all errors."""
        result = (
            L1ObjectManagerError.KEYS_NOT_FOUND
            | L1ObjectManagerError.KEYS_NOT_COMMITTED
            | L1ObjectManagerError.KEYS_ALREADY_LOCKED
        )
        assert result.value == 0x19
        assert L1ObjectManagerError.KEYS_NOT_FOUND in result
        assert L1ObjectManagerError.KEYS_NOT_COMMITTED in result
        assert L1ObjectManagerError.KEYS_ALREADY_LOCKED in result

    def test_bitwise_or_with_success(self):
        """OR-ing with SUCCESS should not change the error."""
        result = L1ObjectManagerError.KEYS_NOT_FOUND | L1ObjectManagerError.SUCCESS
        assert result == L1ObjectManagerError.KEYS_NOT_FOUND

    def test_in_operator_for_flag_check(self):
        """The 'in' operator should work for checking flags in IntFlag."""
        combined = (
            L1ObjectManagerError.KEYS_NOT_FOUND
            | L1ObjectManagerError.KEYS_ALREADY_EXIST
        )
        assert L1ObjectManagerError.KEYS_NOT_FOUND in combined
        assert L1ObjectManagerError.KEYS_ALREADY_EXIST in combined
        assert L1ObjectManagerError.KEYS_NOT_RESERVED not in combined


# =============================================================================
# strerror() Tests
# =============================================================================


class TestStrerror:
    """Tests for strerror() function."""

    # L1MemoryManagerError tests

    def test_strerror_memory_manager_success(self):
        """strerror should return success message for L1MemoryManagerError.SUCCESS."""
        result = strerror(L1MemoryManagerError.SUCCESS)
        assert result == "Operation succeeded."

    def test_strerror_memory_manager_out_of_memory(self):
        """strerror should return OOM message for L1MemoryManagerError.OUT_OF_MEMORY."""
        result = strerror(L1MemoryManagerError.OUT_OF_MEMORY)
        assert result == "Operation failed due to insufficient memory."

    # L1ObjectManagerError single error tests

    def test_strerror_object_manager_success(self):
        """strerror should return success message for L1ObjectManagerError.SUCCESS."""
        result = strerror(L1ObjectManagerError.SUCCESS)
        assert result == "Operation succeeded."

    def test_strerror_keys_not_found(self):
        """strerror should return appropriate message for KEYS_NOT_FOUND."""
        result = strerror(L1ObjectManagerError.KEYS_NOT_FOUND)
        assert "keys not found" in result.lower()

    def test_strerror_keys_already_exist(self):
        """strerror should return appropriate message for KEYS_ALREADY_EXIST."""
        result = strerror(L1ObjectManagerError.KEYS_ALREADY_EXIST)
        assert "keys existed" in result.lower()

    def test_strerror_keys_not_reserved(self):
        """strerror should return appropriate message for KEYS_NOT_RESERVED."""
        result = strerror(L1ObjectManagerError.KEYS_NOT_RESERVED)
        assert "reserved" in result.lower()

    def test_strerror_keys_not_committed(self):
        """strerror should return appropriate message for KEYS_NOT_COMMITTED."""
        result = strerror(L1ObjectManagerError.KEYS_NOT_COMMITTED)
        assert "committed" in result.lower()

    def test_strerror_keys_already_locked(self):
        """strerror should return appropriate message for KEYS_ALREADY_LOCKED."""
        result = strerror(L1ObjectManagerError.KEYS_ALREADY_LOCKED)
        assert "locked" in result.lower()

    # L1ObjectManagerError combined error tests

    def test_strerror_combined_errors(self):
        """strerror should include messages for all combined errors."""
        combined = L1ObjectManagerError.KEYS_NOT_FOUND.mix_error(
            L1ObjectManagerError.KEYS_NOT_COMMITTED
        )
        result = strerror(combined)
        assert "keys not found" in result.lower()
        assert "committed" in result.lower()

    def test_strerror_all_errors_combined(self):
        """strerror should include all error messages when all errors are combined."""
        combined = L1ObjectManagerError.SUCCESS
        combined = combined.mix_error(L1ObjectManagerError.KEYS_NOT_FOUND)
        combined = combined.mix_error(L1ObjectManagerError.KEYS_ALREADY_EXIST)
        combined = combined.mix_error(L1ObjectManagerError.KEYS_NOT_RESERVED)
        combined = combined.mix_error(L1ObjectManagerError.KEYS_NOT_COMMITTED)
        combined = combined.mix_error(L1ObjectManagerError.KEYS_ALREADY_LOCKED)

        result = strerror(combined)
        assert "keys not found" in result.lower()
        assert "keys existed" in result.lower()
        assert "reserved" in result.lower()
        assert "committed" in result.lower()
        assert "locked" in result.lower()

    # Edge case tests

    def test_strerror_unknown_type(self):
        """strerror should return 'Unknown error.' for unsupported types."""
        # Create a mock object that's neither L1MemoryManagerError nor
        # L1ObjectManagerError
        result = strerror("not_an_error")  # type: ignore
        assert result == "Unknown error."
