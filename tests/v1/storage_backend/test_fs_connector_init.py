# SPDX-License-Identifier: Apache-2.0
"""Filesystem initialization must create configured storage directories."""

# Standard
from pathlib import Path
import asyncio
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_allocators.tensor_memory_allocator import TensorMemoryAllocator
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.connector.fs_connector import FSConnector
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


@pytest.mark.no_shared_allocator
@pytest.mark.skipif(not hasattr(os, "statvfs"), reason="POSIX filesystem metadata")
@pytest.mark.parametrize("existing", [True, False], ids=["existing", "new"])
@pytest.mark.parametrize(
    ("use_odirect", "save_chunk_meta"),
    [(False, False), (True, False), (True, True)],
    ids=["buffered", "direct", "metadata_disables_direct"],
)
@pytest.mark.parametrize("path_count", [1, 2], ids=["single_path", "multiple_paths"])
def test_fs_connector_creates_base_paths(
    tmp_path: Path,
    existing: bool,
    use_odirect: bool,
    save_chunk_meta: bool,
    path_count: int,
) -> None:
    """Create new base and temporary directories before filesystem inspection."""
    paths = [tmp_path / f"store-{index}" / "cache" for index in range(path_count)]
    if existing:
        for path in paths:
            path.mkdir(parents=True)
    config = LMCacheEngineConfig.from_defaults(
        chunk_size=2,
        extra_config={
            "remote_storage_plugin.fs.base_path": ",".join(map(str, paths)),
            "fs_connector_relative_tmp_dir": ".tmp",
            "fs_connector_use_odirect": use_odirect,
            "save_chunk_meta": save_chunk_meta,
        },
    )
    metadata = LMCacheMetadata(
        model_name="fs-init-regression",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=torch.uint8,
        kv_shape=(1, 2, 2, 1, 2),
        chunk_size=2,
    )
    allocator = TensorMemoryAllocator(torch.empty(128 * 1024, dtype=torch.uint8))
    backend = LocalCPUBackend(config, metadata, memory_allocator=allocator)
    loop = asyncio.new_event_loop()
    connector = None
    try:
        connector = FSConnector(loop, backend, config, plugin_name="fs")
        assert all(path.is_dir() for path in paths)
        assert all((path / ".tmp").is_dir() for path in paths)
        assert connector.use_odirect is (use_odirect and not save_chunk_meta)
        if connector.use_odirect:
            assert connector.os_disk_bs == os.statvfs(paths[0]).f_bsize
    finally:
        if connector is not None:
            loop.run_until_complete(connector.close())
        loop.close()
        backend.close()
