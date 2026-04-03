# SPDX-License-Identifier: Apache-2.0
# Third Party
import pytest

pytest.importorskip("nixl", reason="nixl package is required for nixl tests")

# First Party
from lmcache.v1.storage_backend.nixl_storage_backend import NixlStorageConfig


class TestNixlMultipath:
    """Test cases for NIXL multipath functionality."""

    def test_validate_nixl_path_single_path(self):
        """Test validate_nixl_path with a single path string."""
        path = "/tmp/nixl/cache"
        path_sharding = "by_gpu"

        result = NixlStorageConfig.validate_nixl_path(path, path_sharding, 0)

        # Should return the same path since there's only one
        assert result == path

    def test_validate_nixl_path_multiple_paths(self):
        """Test validate_nixl_path with a list of paths."""
        paths = ["/tmp/nixl/cache0", "/tmp/nixl/cache1", "/tmp/nixl/cache2"]
        path_sharding = "by_gpu"

        # Test with worker_id 0
        result = NixlStorageConfig.validate_nixl_path(
            paths, path_sharding, 0
        )
        assert result == paths[0]

        # Test with worker_id 1
        result = NixlStorageConfig.validate_nixl_path(
            paths, path_sharding, 1
        )
        assert result == paths[1]

        # Test with worker_id 2
        result = NixlStorageConfig.validate_nixl_path(
            paths, path_sharding, 2
        )
        assert result == paths[2]

        # Test with worker_id 3 (should wrap around to paths[0])
        result = NixlStorageConfig.validate_nixl_path(
            paths, path_sharding, 3
        )
        assert result == paths[0]

    def test_validate_nixl_path_none_path(self):
        """Test validate_nixl_path with None path."""
        with pytest.raises(AssertionError, match="nixl_path cannot be None"):
            NixlStorageConfig.validate_nixl_path(None, "by_gpu", 0)

    def test_validate_nixl_path_empty_list(self):
        """Test validate_nixl_path with empty path list."""
        with pytest.raises(AssertionError, match="nixl_path cannot be an empty list"):
            NixlStorageConfig.validate_nixl_path([], "by_gpu", 0)

    def test_validate_nixl_path_unsupported_sharding(self):
        """Test validate_nixl_path with unsupported path sharding."""
        path = "/tmp/nixl/cache"
        with pytest.raises(AssertionError, match="Unsupported path_sharding"):
            NixlStorageConfig.validate_nixl_path(path, "unsupported_sharding", 0)

    def test_validate_nixl_path_list_conversion(self):
        """Test that string path is properly converted to list."""
        path = "/tmp/nixl/cache"
        path_sharding = "by_gpu"

        # The function should convert string to list internally
        result = NixlStorageConfig.validate_nixl_path(path, path_sharding, 0)
        assert result == path
