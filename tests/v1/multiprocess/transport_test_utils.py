# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for request-transport integration tests."""

# Standard
import importlib.util

# Third Party
import pytest

# The native gRPC server needs pygrpc-native, which is installed separately.
requires_pygrpc_native = pytest.mark.skipif(
    importlib.util.find_spec("pygrpc_native") is None,
    reason="pygrpc-native is not installed",
)
