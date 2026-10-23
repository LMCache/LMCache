# SPDX-License-Identifier: Apache-2.0
"""Memory tags of L1 managers and L2 adapters, and L1 affinity resolution."""

# Standard
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemoryResidency, MemorySegment, MemoryTags
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    parse_args,
)
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.nixl_native_l2_adapter import (
    nixl_backend_memory_tags,
)
from lmcache.v1.distributed.storage_controllers.utils import resolve_l1_affinity

pytestmark = pytest.mark.no_shared_allocator

DRAM = MemoryTags(MemorySegment.DRAM, MemoryResidency.LOCAL)
CXL = MemoryTags(MemorySegment.CXL, MemoryResidency.LOCAL)
GDS = MemoryTags(MemorySegment.GDS, MemoryResidency.LOCAL)
LOCAL_STORAGE = MemoryTags(MemorySegment.STORAGE, MemoryResidency.LOCAL)
REMOTE_STORAGE = MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE)


class _TaggedL1:
    """Minimal L1 stand-in exposing only its memory tags."""

    def __init__(self, tags: MemoryTags) -> None:
        self._tags = tags

    def get_memory_tags(self) -> MemoryTags:
        return self._tags


def _l1(tags: MemoryTags) -> L1ManagerInterface:
    return cast(L1ManagerInterface, _TaggedL1(tags))


class _CxlFirstConfig(MockL2AdapterConfig):
    """Mock adapter config that prefers a CXL L1 over a DRAM one."""

    def get_wanted_l1_memory_tags(self) -> tuple[MemoryTags, ...]:
        return (CXL, DRAM)


def _mock_config() -> MockL2AdapterConfig:
    return MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=10)


def _l2_config(spec: dict[str, Any]) -> L2AdapterConfigBase:
    args = ["--eviction-policy", "LRU", "--l1-size-gb", "1"]
    config = parse_args([*args, "--l2-adapter", json.dumps(spec)])
    return config.l2_adapter_config.adapters[0]


@contextmanager
def _adapter(spec: dict[str, Any]) -> Iterator[L2AdapterInterface]:
    adapter = create_l2_adapter(_l2_config(spec), None)
    try:
        yield adapter
    finally:
        adapter.close()


def _devdax_file(tmp_path: Path) -> str:
    path = tmp_path / "dax"
    path.write_bytes(b"\0" * (1 << 20))
    return str(path)


# ---------------------------------------------------------------------------
# Affinity resolution
# ---------------------------------------------------------------------------


def test_resolve_picks_first_matching_l1_in_placement_order() -> None:
    l1s = {"cxl": _l1(CXL), "first": _l1(DRAM), "second": _l1(DRAM)}
    assert resolve_l1_affinity(_mock_config(), l1s) == "first"


def test_resolve_follows_wanted_preference_order() -> None:
    config = _CxlFirstConfig(max_size_gb=1, mock_bandwidth_gb=10)
    assert resolve_l1_affinity(config, {"dram": _l1(DRAM)}) == "dram"
    l1s = {"dram": _l1(DRAM), "cxl": _l1(CXL)}
    assert resolve_l1_affinity(config, l1s) == "cxl"


def test_resolve_raises_when_no_l1_has_wanted_tags() -> None:
    with pytest.raises(ValueError, match="No L1 has the memory tags.*'mock'"):
        resolve_l1_affinity(_mock_config(), {"gds": _l1(GDS)})


def test_default_wanted_l1_tags_prefer_dram_over_cxl() -> None:
    assert _mock_config().get_wanted_l1_memory_tags() == (DRAM, CXL)
    assert resolve_l1_affinity(_mock_config(), {"cxl": _l1(CXL)}) == "cxl"


def test_fault_inject_wants_inner_l1_tags() -> None:
    config = _l2_config(
        {
            "type": "fault_inject",
            "inner": {"type": "mock", "max_size_gb": 1, "mock_bandwidth_gb": 10},
        }
    )
    assert config.get_wanted_l1_memory_tags() == (DRAM, CXL)


# ---------------------------------------------------------------------------
# L1 manager tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "devdax_size_in_bytes, expected",
    [(None, DRAM), (0, CXL), (1 << 20, DRAM)],
    ids=["dram", "pure-devdax", "hybrid-dram-devdax"],
)
def test_l1_manager_tags(
    tmp_path: Path, devdax_size_in_bytes: int | None, expected: MemoryTags
) -> None:
    devdax: dict[str, Any] = {}
    if devdax_size_in_bytes is not None:
        devdax = {
            "devdax_path": _devdax_file(tmp_path),
            "devdax_size_in_bytes": devdax_size_in_bytes,
        }
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=1 << 20, use_lazy=False, shm_name="", **devdax
            ),
            eviction=EvictionConfig(eviction_policy="LRU"),
        )
    )
    try:
        assert manager.get_memory_tags() == expected
    finally:
        manager.close()


# ---------------------------------------------------------------------------
# L2 adapter tags
# ---------------------------------------------------------------------------


def test_mock_adapter_is_local_storage() -> None:
    with _adapter({"type": "mock", "max_size_gb": 1, "mock_bandwidth_gb": 10}) as a:
        assert a.get_memory_tags() == LOCAL_STORAGE


@pytest.mark.parametrize(
    "shared, expected", [(False, LOCAL_STORAGE), (True, REMOTE_STORAGE)]
)
def test_fs_adapter_is_remote_only_when_shared(
    tmp_path: Path, shared: bool, expected: MemoryTags
) -> None:
    spec = {"type": "fs", "base_path": str(tmp_path), "shared": shared}
    with _adapter(spec) as adapter:
        assert adapter.get_memory_tags() == expected


def test_fault_inject_adapter_reports_shared_inner_as_remote(tmp_path: Path) -> None:
    """A shared inner adapter stays remote behind the fault-inject wrapper."""
    spec = {
        "type": "fault_inject",
        "inner": {"type": "fs", "base_path": str(tmp_path), "shared": True},
    }
    with _adapter(spec) as adapter:
        assert adapter.get_memory_tags() == REMOTE_STORAGE


def test_fault_inject_adapter_forwards_inner_tags() -> None:
    spec = {
        "type": "fault_inject",
        "inner": {"type": "mock", "max_size_gb": 1, "mock_bandwidth_gb": 10},
    }
    with _adapter(spec) as adapter:
        assert adapter.get_memory_tags() == LOCAL_STORAGE


@pytest.mark.parametrize(
    "backend, shared, expected",
    [
        ("GDS", False, LOCAL_STORAGE),
        ("GDS_MT", False, LOCAL_STORAGE),
        ("POSIX", False, LOCAL_STORAGE),
        ("POSIX", True, REMOTE_STORAGE),
        ("GDS", True, REMOTE_STORAGE),
        ("HF3FS", False, REMOTE_STORAGE),
        ("OBJ", False, REMOTE_STORAGE),
    ],
)
def test_nixl_backend_memory_tags(
    backend: str, shared: bool, expected: MemoryTags
) -> None:
    assert nixl_backend_memory_tags(backend, shared) == expected
