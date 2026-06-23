# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`lmcache.v1.storage_backend.path_sharder`."""

# Standard
from unittest.mock import patch
import os
import shutil
import tempfile

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.path_sharder import PathSharder


def _make_key(chunk_hash: int) -> CacheEngineKey:
    """Build a minimal CacheEngineKey with the given chunk_hash."""
    return CacheEngineKey(
        model_name="m",
        world_size=1,
        worker_id=0,
        chunk_hash=chunk_hash,
        dtype=torch.float16,
    )


class TestPathSharder:
    """Tests for PathSharder class."""

    def test_single_path(self):
        d = tempfile.mkdtemp()
        try:
            s = PathSharder(d, strategy="by_gpu", dst_device="cuda:0")
            assert s.selected == d
            assert s.all_paths == [d]
            assert s.strategy == "by_gpu"
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_multi_path_selects_by_device_id(self):
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            for i, d in enumerate(dirs):
                s = PathSharder(csv, strategy="by_gpu", dst_device=f"cuda:{i}")
                assert s.selected == d
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_modulo_wraps(self):
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:4")
            # 4 % 2 == 0
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_create_dirs(self):
        base = tempfile.mkdtemp()
        try:
            paths = [os.path.join(base, f"nvme{i}") for i in range(3)]
            csv = ",".join(paths)
            PathSharder(csv, strategy="by_gpu", dst_device="cuda:0", create_dirs=True)
            for p in paths:
                assert os.path.isdir(p)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_no_create_dirs_by_default(self):
        base = tempfile.mkdtemp()
        try:
            new_dir = os.path.join(base, "should_not_exist")
            PathSharder(new_dir, strategy="by_gpu", dst_device="cuda:0")
            assert not os.path.exists(new_dir)
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_empty_csv_raises(self):
        with pytest.raises(ValueError, match="At least one path"):
            PathSharder("", strategy="by_gpu", dst_device="cuda:0")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="At least one path"):
            PathSharder("  , ,  ", strategy="by_gpu", dst_device="cuda:0")

    def test_unsupported_strategy_raises(self):
        d = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError, match="Unsupported path sharding"):
                PathSharder(d, strategy="round_robin", dst_device="cuda:0")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_strips_whitespace(self):
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = f"  {dirs[0]}  ,  {dirs[1]}  "
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:0")
            assert s.all_paths == dirs
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_normalizes_trailing_slashes(self):
        d = tempfile.mkdtemp()
        try:
            s = PathSharder(f"{d}/", strategy="by_gpu", dst_device="cuda:0")
            assert s.selected == d
            assert s.all_paths == [d]
            assert s.assigned_paths == [d]
        finally:
            shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=False,
    )
    def test_cpu_device_selects_first_path(self, _avail):
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cpu")
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_all_paths_returns_copy(self):
        d = tempfile.mkdtemp()
        try:
            s = PathSharder(d, strategy="by_gpu", dst_device="cuda:0")
            paths = s.all_paths
            paths.append("/rogue")
            assert "/rogue" not in s.all_paths
        finally:
            shutil.rmtree(d, ignore_errors=True)

    # -- device-resolution edge cases (exercised via public API) -----------

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=True,
    )
    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.current_device",
        return_value=1,
    )
    def test_bare_cuda_uses_current_device(self, _cur, _avail):
        """Bare 'cuda' resolves to torch_dev.current_device()."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda")
            assert s.selected == dirs[1]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=False,
    )
    def test_bare_cuda_no_gpu_selects_first(self, _avail):
        """Bare 'cuda' with no GPU falls back to device 0."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda")
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=True,
    )
    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.current_device",
        return_value=2,
    )
    def test_cpu_device_always_selects_first(self, _cur, _avail):
        """'cpu' always resolves to index 0, even when CUDA is available."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cpu")
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=True,
    )
    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.current_device",
        return_value=2,
    )
    def test_malformed_device_empty_index_falls_back(self, _cur, _avail):
        """'cuda:' (no int) falls back to current_device."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:")
            assert s.selected == dirs[2]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    @patch(
        "lmcache.v1.storage_backend.path_sharder.torch_dev.is_available",
        return_value=False,
    )
    def test_malformed_device_non_numeric_falls_back(self, _avail):
        """'cuda:foo' falls back to 0 when CUDA is unavailable."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:foo")
            assert s.selected == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)


class TestPathSharderLegacyBranch:
    """Guard the metadata-absent branch used by GDS and any legacy caller.

    This branch must stay a pure ``paths[device_id % len(paths)]`` mapping
    with no new validation, so that backends which do not pass local-rank
    metadata are unaffected by worker-aware assignment.
    """

    def test_device_id_beyond_num_paths_no_error(self):
        """device_id >= num_paths must wrap via modulo, never raise."""
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            # 5 % 2 == 1, non-multiple device id, no metadata -> no error.
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:5")
            assert s.selected == dirs[1]
            assert s.assigned_paths == [dirs[1]]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_assigned_paths_single_without_metadata(self):
        """Without metadata, exactly one path is assigned."""
        dirs = [tempfile.mkdtemp() for _ in range(4)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(csv, strategy="by_gpu", dst_device="cuda:2")
            assert s.assigned_paths == [dirs[2]]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_partial_metadata_raises(self):
        """Providing only one of local_rank / local_world_size is an error."""
        d = tempfile.mkdtemp()
        try:
            with pytest.raises(ValueError, match="must be provided together"):
                PathSharder(d, strategy="by_gpu", dst_device="cuda:0", local_rank=0)
            with pytest.raises(ValueError, match="must be provided together"):
                PathSharder(
                    d,
                    strategy="by_gpu",
                    dst_device="cuda:0",
                    local_world_size=1,
                )
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestPathSharderWorkerAware:
    """Tests for the metadata-present worker-aware assignment branch."""

    def test_one_to_one_workers_equal_paths(self):
        """workers == paths -> each rank owns exactly one path."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            for r in range(3):
                s = PathSharder(
                    csv,
                    strategy="by_gpu",
                    dst_device=f"cuda:{r}",
                    local_rank=r,
                    local_world_size=3,
                )
                assert s.assigned_paths == [dirs[r]]
                assert s.selected == dirs[r]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_workers_more_than_paths_shares(self):
        """workers > paths (exact multiple) -> modulo sharing."""
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            # 4 workers, 2 paths: rank 2 -> paths[2 % 2] == paths[0]
            s = PathSharder(
                csv,
                strategy="by_gpu",
                dst_device="cuda:2",
                local_rank=2,
                local_world_size=4,
            )
            assert s.assigned_paths == [dirs[0]]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_workers_fewer_than_paths_subset(self):
        """workers < paths -> contiguous subset per worker."""
        dirs = [tempfile.mkdtemp() for _ in range(4)]
        try:
            csv = ",".join(dirs)
            s0 = PathSharder(
                csv,
                strategy="by_gpu",
                dst_device="cuda:0",
                local_rank=0,
                local_world_size=2,
            )
            s1 = PathSharder(
                csv,
                strategy="by_gpu",
                dst_device="cuda:1",
                local_rank=1,
                local_world_size=2,
            )
            assert s0.assigned_paths == dirs[0:2]
            assert s1.assigned_paths == dirs[2:4]
            assert s0.selected == dirs[0]
            assert s1.selected == dirs[2]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_non_divisible_layout_raises(self):
        """3 paths, 2 workers is neither equal nor an exact multiple."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            with pytest.raises(ValueError, match="exact multiples"):
                PathSharder(
                    csv,
                    strategy="by_gpu",
                    dst_device="cuda:0",
                    local_rank=0,
                    local_world_size=2,
                )
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_workers_more_than_paths_non_multiple_raises(self):
        """3 workers, 2 paths is not an exact multiple."""
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            with pytest.raises(ValueError, match="exact multiples"):
                PathSharder(
                    csv,
                    strategy="by_gpu",
                    dst_device="cuda:0",
                    local_rank=0,
                    local_world_size=3,
                )
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_rank_out_of_range_raises(self):
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            with pytest.raises(ValueError, match="range"):
                PathSharder(
                    csv,
                    strategy="by_gpu",
                    dst_device="cuda:0",
                    local_rank=2,
                    local_world_size=2,
                )
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_non_positive_world_size_raises(self):
        dirs = [tempfile.mkdtemp() for _ in range(2)]
        try:
            csv = ",".join(dirs)
            with pytest.raises(ValueError, match="positive"):
                PathSharder(
                    csv,
                    strategy="by_gpu",
                    dst_device="cuda:0",
                    local_rank=0,
                    local_world_size=0,
                )
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)


class TestResolvePathForKey:
    """Tests for per-key path routing within a worker's assigned subset."""

    def test_single_path_skips_hashing(self):
        """A single-path worker always returns that path."""
        dirs = [tempfile.mkdtemp() for _ in range(3)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(
                csv,
                strategy="by_gpu",
                dst_device="cuda:0",
                local_rank=0,
                local_world_size=3,
            )
            for h in (0, 1, 12345, 2**63):
                assert s.resolve_path_for_key(_make_key(h)) == dirs[0]
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)

    def test_subset_routing_deterministic_and_in_subset(self):
        """Same key -> same path; result always within the assigned subset."""
        dirs = [tempfile.mkdtemp() for _ in range(4)]
        try:
            csv = ",".join(dirs)
            s = PathSharder(
                csv,
                strategy="by_gpu",
                dst_device="cuda:0",
                local_rank=0,
                local_world_size=2,
            )
            subset = dirs[0:2]
            seen = set()
            for h in range(100):
                key = _make_key(h)
                p1 = s.resolve_path_for_key(key)
                p2 = s.resolve_path_for_key(key)
                assert p1 == p2
                assert p1 in subset
                seen.add(p1)
            # Over 100 distinct hashes both subset paths should be used.
            assert seen == set(subset)
        finally:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
