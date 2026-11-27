# SPDX-License-Identifier: Apache-2.0
"""Plugin L1 tier: ``--l1-manager`` type ``PLUGIN`` and the factory's loading.

The plugin is :mod:`tests.v1.distributed.plugin_l1_fixture`.
"""

# Standard
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemorySegment
from lmcache.v1.distributed.config import (
    PluginL1Config,
    PluginL1ManagerConfig,
    get_configured_capacity_bytes,
    l1_exposes_single_memory_region,
    l1_medium_name,
    parse_args,
)
from lmcache.v1.distributed.l1_manager.factory import create_l1_manager
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_controllers.utils import resolve_l1_affinity
from tests.v1.distributed.plugin_l1_fixture import HostL1Manager

FIXTURE = "tests.v1.distributed.plugin_l1_fixture"
SIZE_GB = 1 / 16
PLUGIN = {
    "type": "PLUGIN",
    "tag": "host",
    "size_gb": SIZE_GB,
    "module_path": FIXTURE,
    "class_name": "HostL1Manager",
}


def _args(specs: list[dict[str, object]]) -> list[str]:
    return [
        "--eviction-policy",
        "LRU",
        *[item for spec in specs for item in ("--l1-manager", json.dumps(spec))],
    ]


def test_plugin_json_is_parsed_without_importing_the_module() -> None:
    spec = {**PLUGIN, "module_path": "no.such.module", "params": {"note": "x"}}
    config = parse_args(_args([spec]))
    (l1,) = config.l1_manager_configs
    assert isinstance(l1, PluginL1ManagerConfig)
    size = int(SIZE_GB * (1 << 30))
    assert l1.plugin_l1_config == PluginL1Config(
        module_path="no.such.module",
        class_name="HostL1Manager",
        size_in_bytes=size,
        params={"note": "x"},
    )
    assert l1.tag == "host"
    assert get_configured_capacity_bytes(l1) == size
    assert l1_medium_name(l1) == L1BackendType.PLUGIN.value
    assert not l1_exposes_single_memory_region(config)


@pytest.mark.parametrize(
    "spec, error",
    [
        ({**PLUGIN, "module_path": ""}, "module_path must be a non-empty string"),
        ({**PLUGIN, "class_name": 3}, "class_name must be a non-empty string"),
        ({**PLUGIN, "params": [1]}, "params must be an object"),
        ({**PLUGIN, "path": "/x"}, "Unknown L1 fields"),
        ({**PLUGIN, "tag": "_default"}, "Only DRAM"),
        ({**PLUGIN, "type": "NVME"}, "DRAM, DEVDAX, GDS, PLUGIN or SHADOW"),
    ],
)
def test_invalid_plugin_json(spec: dict[str, object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_args(_args([spec]))


def test_factory_builds_the_named_class() -> None:
    (l1,) = parse_args(_args([PLUGIN])).l1_manager_configs
    manager = create_l1_manager(l1)
    try:
        assert isinstance(manager, HostL1Manager)
        assert manager.backend_type == L1BackendType.PLUGIN
        assert manager.get_memory_tags().segment == MemorySegment.DRAM
        assert manager.get_capacity_bytes() == int(SIZE_GB * (1 << 30))
    finally:
        manager.close()


def test_l2_adapters_bind_by_the_plugins_memory_tags() -> None:
    """A plugin joins L1-L2 affinity through its tags, like a built-in L1."""
    (l1,) = parse_args(_args([PLUGIN])).l1_manager_configs
    manager = create_l1_manager(l1)
    try:
        adapter = MockL2AdapterConfig(max_size_gb=0.1, mock_bandwidth_gb=1.0)
        assert resolve_l1_affinity(adapter, {"host": manager}) == "host"
    finally:
        manager.close()


def test_plugin_validates_its_own_params() -> None:
    (l1,) = parse_args(_args([{**PLUGIN, "params": {"bogus": 1}}])).l1_manager_configs
    with pytest.raises(ValueError, match="Unknown HostL1Manager params"):
        create_l1_manager(l1)


@pytest.mark.parametrize(
    "module_path, class_name, error, match",
    [
        ("no.such.module", "X", ImportError, "Could not import L1 plugin module"),
        (FIXTURE, "Missing", ValueError, "not a BaseL1Manager subclass"),
        (FIXTURE, "NotAManager", ValueError, "not a BaseL1Manager subclass"),
    ],
)
def test_factory_rejects_a_bad_plugin(
    module_path: str, class_name: str, error: type[Exception], match: str
) -> None:
    spec = {**PLUGIN, "module_path": module_path, "class_name": class_name}
    (l1,) = parse_args(_args([spec])).l1_manager_configs
    with pytest.raises(error, match=match):
        create_l1_manager(l1)
