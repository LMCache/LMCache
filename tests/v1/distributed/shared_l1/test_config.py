# SPDX-License-Identifier: Apache-2.0
"""Validation of the ``shared`` object of a Device-DAX L1 and of server modes."""

# Standard
from dataclasses import replace
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import MemorySegment
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    EvictionConfig,
    L1MemoryManagerConfig,
    ShadowL1ManagerConfig,
    SharedDevDaxConfig,
    StorageManagerConfig,
    parse_args,
)
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    P2PConfig,
    RuntimePluginConfig,
)
from lmcache.v1.multiprocess.server import check_shared_devdax_support

SHARED = {"orchestrator": "h:1", "region_id": "region", "client_id": "host"}
PRIVATE = dict(type="DEVDAX", tag="pool", size_gb=1, path="/dev/dax0.0", offset_gb=1)
SPEC = {**PRIVATE, "shared": SHARED}
DRAM = {"type": "DRAM", "tag": "_default", "size_gb": 1}
SHADOW: dict[str, object] = {"type": "SHADOW", "tag": "cxl-views", "segment": "cxl"}
MEMORY = L1MemoryManagerConfig(
    1 << 30, use_lazy=False, shm_name="", devdax_path="/dev/dax0.0"
)


def parse(*specs: dict[str, object]) -> StorageManagerConfig:
    """Parse ``--l1-manager`` specs under a global LRU eviction default."""
    flags = [arg for spec in specs for arg in ("--l1-manager", json.dumps(spec))]
    return parse_args(["--eviction-policy", "LRU", *flags])


def test_shared_object_is_optional_unique_and_reuses_the_dax_fields() -> None:
    config = parse(SPEC, DRAM, SHADOW)
    shared, _, shadow = config.l1_manager_configs
    private = parse(PRIVATE).l1_manager_configs[0]
    assert isinstance(shadow, ShadowL1ManagerConfig)
    assert shadow.segment is MemorySegment.CXL
    assert shadow.memory_config.size_in_bytes == 0
    assert isinstance(shared, DevDaxL1ManagerConfig)
    assert shared.shared == SharedDevDaxConfig("h:1", "region", "host", 5.0)
    assert shared.memory_config.devdax_path == "/dev/dax0.0"
    assert shared.memory_config.devdax_offset_in_bytes == 1 << 30
    # The global LRU default reaches the private L1 but not the shared one.
    assert shared.eviction is not None and shared.eviction.eviction_policy == "noop"
    assert isinstance(private, DevDaxL1ManagerConfig) and private.shared is None
    assert private.eviction is not None and private.eviction.eviction_policy == "LRU"
    with pytest.raises(ValueError, match="Only one shared"):
        parse(SPEC, {**SPEC, "tag": "peer", "offset_gb": 3})


@pytest.mark.parametrize(
    "change",
    [
        {"shared": {**SHARED, "orchestrator": ""}},
        {"shared": {**SHARED, "region_id": None}},
        {"shared": {**SHARED, "client_id": " "}},
        {"shared": {**SHARED, "rpc_timeout_seconds": 0}},
        {"shared": {**SHARED, "rpc_timeout_seconds": float("inf")}},
        {"shared": {**SHARED, "rpc_timeout_seconds": True}},
        {"shared": {**SHARED, "path": "/dev/dax0.0"}},
        {"shared": {**SHARED, "unknown": 1}},
        {"shared": {"orchestrator": "h:1", "region_id": "r"}},
        {"shared": "h:1"},
        {"type": "DRAM"},
        {"type": "GDS"},
        {"eviction": {"eviction_policy": "LRU"}},
        {"eviction": {"trigger_watermark": 0.5}},
        {"path": None},
    ],
)
def test_invalid_shared_settings(change: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parse({**SPEC, **change})


@pytest.mark.parametrize(
    "eviction,memory,timeout",
    [
        # None inherits the global LRU policy, which a shared L1 cannot use.
        (None, MEMORY, 5.0),
        (EvictionConfig("ARC"), MEMORY, 5.0),
        (EvictionConfig("noop"), MEMORY, float("nan")),
    ],
)
def test_programmatic_construction_is_validated(
    eviction: EvictionConfig | None, memory: L1MemoryManagerConfig, timeout: float
) -> None:
    with pytest.raises(ValueError):
        shared = SharedDevDaxConfig("h:1", "region", "host", timeout)
        config = DevDaxL1ManagerConfig(
            memory, tag="pool", eviction=eviction, shared=shared
        )
        StorageManagerConfig(config, EvictionConfig("LRU"), l1_manager_configs=[config])


BASE_MP = MPServerConfig(supported_transfer_mode="lmcache_driven")


@pytest.mark.parametrize(
    "mp_config",
    [
        replace(BASE_MP, supported_transfer_mode="auto"),
        replace(BASE_MP, supported_transfer_mode="engine_driven"),
        replace(BASE_MP, engine_type="blend"),
        replace(BASE_MP, p2p_config=P2PConfig(advertise_url="h:9000")),
        replace(BASE_MP, runtime_plugin_config=RuntimePluginConfig(["plugin.py"])),
    ],
)
def test_unsupported_server_modes(mp_config: MPServerConfig) -> None:
    with pytest.raises(ValueError, match="shared Device-DAX"):
        check_shared_devdax_support(mp_config, parse(SPEC))


def test_supported_server_modes() -> None:
    check_shared_devdax_support(BASE_MP, parse(SPEC, DRAM))
    # Without a shared L1 the server keeps every mode.
    every_mode = replace(BASE_MP, supported_transfer_mode="auto", engine_type="blend")
    check_shared_devdax_support(every_mode, parse(PRIVATE))
