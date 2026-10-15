# SPDX-License-Identifier: Apache-2.0
"""Actual CUDA copies and fs L2 round-trip with externally loaded selectors."""

# Standard
from pathlib import Path
import argparse
import importlib
import json
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.test_external_policies import (
    external_package as external_package,
)
from tests.v1.distributed.utils import single_row_spec

pytestmark = [pytest.mark.cuda, pytest.mark.no_shared_allocator]


@pytest.mark.parametrize("store_class", ["ExternalStore", "ExternalStoreAll"])
def test_external_selector_cuda_fs_round_trip(
    external_package: str,  # noqa: F811 - shared pytest fixture imported above
    tmp_path: Path,
    store_class: str,
) -> None:
    """GPU data stays usable when L2 is suppressed or reloaded from real files."""
    if not torch.cuda.is_available():
        pytest.skip("Requires an actual CUDA device")
    parser = argparse.ArgumentParser()
    add_storage_manager_args(parser)
    cache_path = tmp_path / "cache"
    config = parse_args_to_config(
        parser.parse_args(
            [
                "--l1-size-gb",
                "0.0625",
                "--no-l1-use-lazy",
                "--eviction-policy",
                "noop",
                "--l2-store-policy",
                f"{external_package}:{store_class}",
                "--l2-prefetch-policy",
                f"{external_package}:ExternalPrefetch",
                "--l2-adapter",
                json.dumps({"type": "fs", "base_path": str(cache_path)}),
            ]
        )
    )
    module = importlib.import_module(external_package)
    manager = StorageManager(config)
    try:
        key = ObjectKey(b"cuda-external-policy", "cuda-policy-test", 0)
        layout = MemoryLayoutDesc([torch.Size([4096])], [torch.float32])
        expected = torch.arange(4096, dtype=torch.float32, device="cuda")
        objects = manager.reserve_write([key], layout)
        assert key in objects
        staging = objects[key].tensor
        assert staging is not None
        staging.copy_(expected, non_blocking=True)
        torch.cuda.synchronize()
        manager.finish_write([key])

        deadline = time.monotonic() + 15
        while getattr(module, store_class).calls == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert getattr(module, store_class).calls > 0
        if store_class == "ExternalStoreAll":
            while not list(cache_path.glob("*.data")) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert list(cache_path.glob("*.data")), "L2 did not store actual data"
            deleted = (0, 1)
            while deleted != (1, 0) and time.monotonic() < deadline:
                deleted = manager.delete_l1_keys([key])
                if deleted != (1, 0):
                    time.sleep(0.01)
            assert deleted == (1, 0), "L1 copy stayed locked after completed store"
        else:
            assert list(cache_path.glob("*.data")) == []

        handle = manager.submit_prefetch_task(single_row_spec([key], layout))
        deadline = time.monotonic() + 15
        result = None
        while result is None and time.monotonic() < deadline:
            result = manager.query_prefetch_status(handle)
            if result is None:
                time.sleep(0.01)
        assert result is not None
        assert result.hit_cells[0].popcount() == 1
        assert module.ExternalPrefetch.calls > 0
        with manager.read_prefetched_results([key], result.l1_owners) as read_objects:
            assert read_objects is not None
            source = read_objects[0].tensor
            assert source is not None
            restored = torch.empty_like(expected)
            restored.copy_(source, non_blocking=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        manager.finish_read_prefetched([key], l1_owners=result.l1_owners)
        assert manager.delete_l1_keys([key]) == (1, 0)
        assert manager.report_status()["is_healthy"]
    finally:
        manager.close()
