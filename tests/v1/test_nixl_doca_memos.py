# SPDX-License-Identifier: Apache-2.0
"""Tests for DOCA_MEMOS backend config validation."""

# Third Party
import pytest

pytest.importorskip("nixl")

# First Party
from lmcache.v1.storage_backend.nixl_storage_backend import NixlStorageConfig


class TestValidateNixlBackend:
    """NixlStorageConfig.validate_nixl_backend — the config-acceptance path."""

    def test_doca_memos_cpu_is_valid(self) -> None:
        assert NixlStorageConfig.validate_nixl_backend("DOCA_MEMOS", "cpu") is True

    def test_doca_memos_cuda_is_rejected(self) -> None:
        assert NixlStorageConfig.validate_nixl_backend("DOCA_MEMOS", "cuda") is False

    def test_doca_memos_cuda_with_index_is_rejected(self) -> None:
        # device strings may carry an index suffix (e.g. "cuda:0")
        assert NixlStorageConfig.validate_nixl_backend("DOCA_MEMOS", "cuda:0") is False
