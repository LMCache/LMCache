# SPDX-License-Identifier: Apache-2.0
"""Helpers for shared Device-DAX L1 tests."""

# Standard
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
import json

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.config import DevDaxL1ManagerConfig, parse_args
from lmcache.v1.memory_orchestrator.api import DEFAULT_LAYOUT_FINGERPRINT, RegionUsage
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from tests.v1.memory_orchestrator.harness import REGION_ID

CAPACITY = 128 * 4096


def region_file(path: Path, size_bytes: int) -> Path:
    """Create a zero-filled file standing in for a Device-DAX device.

    The ``DEVDAX`` option maps a regular file through the same code path, so
    CPU tests share a region by mapping one file.
    """
    with open(path, "wb") as region:
        region.truncate(size_bytes)
    return path


def key(index: int) -> ObjectKey:
    """Return a deterministic key for lifecycle and placement tests."""
    return ObjectKey(ObjectKey.IntHash2Bytes(index), "shared-test", 0)


@contextmanager
def region_usage(endpoint: str) -> Iterator[Callable[[], RegionUsage]]:
    """Read the region's counters through a separate probe client."""
    with closing(MemoryOrchestratorClient(endpoint, REGION_ID, "probe")) as client:
        contract = client.describe_region()
        client.register(
            layout_fingerprint=DEFAULT_LAYOUT_FINGERPRINT,
            mapped_bytes=contract.capacity_bytes,
            visibility_mode=contract.visibility_mode,
        )
        yield client.usage


def shared_config(
    region_path: Path,
    endpoint: str,
    client_id: str,
    *,
    size_bytes: int = CAPACITY,
    offset_bytes: int = 0,
    tag: str = "pool",
    rpc_timeout_seconds: float = 5.0,
    region_id: str = REGION_ID,
) -> DevDaxL1ManagerConfig:
    """Parse a shared Device-DAX L1 the way ``--l1-manager`` does."""
    spec = {
        "type": "DEVDAX",
        "tag": tag,
        "size_gb": size_bytes / (1 << 30),
        "offset_gb": offset_bytes / (1 << 30),
        "path": str(region_path),
        "shared": {
            "orchestrator": endpoint,
            "region_id": region_id,
            "client_id": client_id,
            "rpc_timeout_seconds": rpc_timeout_seconds,
        },
    }
    args = ["--eviction-policy", "noop", "--l1-manager", json.dumps(spec)]
    config = parse_args(args).l1_manager_configs[0]
    assert isinstance(config, DevDaxL1ManagerConfig)
    return config
