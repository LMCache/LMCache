# SPDX-License-Identifier: Apache-2.0
"""Plugin L1 tier: ``--l1-manager`` type ``PLUGIN``, the factory's loading, and
the generic transfer path's handling of ``DeviceStagedMemoryObj``.

The plugin is :mod:`tests.v1.distributed.device_staged_l1_fixture`.
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
from lmcache.v1.memory_management import DeviceStagedMemoryObj
from tests.v1.distributed.device_staged_l1_fixture import (
    HostCopyL1Manager,
    HostCopyObject,
)

FIXTURE = "tests.v1.distributed.device_staged_l1_fixture"
PLUGIN = {
    "type": "PLUGIN",
    "tag": "nvme",
    "size_gb": 1,
    "module_path": FIXTURE,
    "class_name": "HostCopyL1Manager",
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
    assert l1.plugin_l1_config == PluginL1Config(
        module_path="no.such.module",
        class_name="HostCopyL1Manager",
        size_in_bytes=1 << 30,
        params={"note": "x"},
    )
    assert l1.tag == "nvme"
    assert get_configured_capacity_bytes(l1) == 1 << 30
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
        assert isinstance(manager, HostCopyL1Manager)
        assert manager.backend_type == L1BackendType.PLUGIN
        assert manager.get_memory_tags().segment == MemorySegment.STORAGE
        assert manager.get_capacity_bytes() == 1 << 30
    finally:
        manager.close()


def test_plugin_validates_its_own_params() -> None:
    (l1,) = parse_args(_args([{**PLUGIN, "params": {"bogus": 1}}])).l1_manager_configs
    with pytest.raises(ValueError, match="Unknown HostCopyL1Manager params"):
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


def test_device_staged_object_is_a_placeholder() -> None:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from tests.v1.distributed.device_staged_l1_fixture import BudgetMemoryManager

    layout = MemoryLayoutDesc([torch.Size([2, 4, 256, 512])], [torch.bfloat16])
    memory = BudgetMemoryManager(1 << 30, 4096)
    _, (obj,) = memory.allocate(layout, 1)
    assert isinstance(obj, DeviceStagedMemoryObj) and isinstance(obj, HostCopyObject)
    assert obj.tensor is None and obj.raw_tensor is None
    assert obj.get_physical_size() == 2 * 4 * 256 * 512 * 2
    assert obj.get_shapes() == [torch.Size([2, 4, 256, 512])]
    with pytest.raises(NotImplementedError, match="device-staged"):
        obj.data_ptr  # noqa: B018
    with pytest.raises(ValueError, match="nothing to retrieve"):
        obj.transfer_through_device(torch.device("cpu"), False, lambda _: None)
    memory.free([obj])
    assert not obj.is_valid()


def _staged_object(nbytes: int) -> HostCopyObject:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.distributed.api import MemoryLayoutDesc
    from tests.v1.distributed.device_staged_l1_fixture import BudgetMemoryManager

    layout = MemoryLayoutDesc([torch.Size([nbytes // 2])], [torch.bfloat16])
    _, (obj,) = BudgetMemoryManager(1 << 30, 4096).allocate(layout, 1)
    assert isinstance(obj, HostCopyObject)
    return obj


def test_generic_path_copies_through_the_lent_buffer() -> None:
    """``gpu_ops`` stores a staged chunk into, and retrieves it from, the
    object's lent buffer; the object sees one store, then one retrieve."""
    # Third Party
    import torch

    # First Party
    from lmcache.v1.gpu_connector.gpu_ops import (
        lmcache_memcpy_async_d2h,
        lmcache_memcpy_async_h2d,
    )

    nbytes = 3 * 4096 + 512  # physical size rounds up past the chunk
    obj = _staged_object(nbytes)
    assert obj.get_physical_size() > nbytes
    chunk = torch.randint(0, 256, (nbytes,), dtype=torch.uint8)

    lmcache_memcpy_async_d2h(chunk.clone(), obj)
    assert obj.transfers == [True]
    assert obj.saved is not None and torch.equal(obj.saved[:nbytes], chunk)

    restored = torch.zeros(nbytes, dtype=torch.uint8)
    lmcache_memcpy_async_h2d(obj, restored)
    assert obj.transfers == [True, False]
    assert torch.equal(restored, chunk)


def test_generic_path_rejects_a_size_mismatch() -> None:
    # Third Party
    import torch

    # First Party
    from lmcache.v1.gpu_connector.gpu_ops import lmcache_memcpy_async_d2h

    obj = _staged_object(4096)
    with pytest.raises(ValueError, match="Size mismatch"):
        lmcache_memcpy_async_d2h(torch.zeros(2048, dtype=torch.uint8), obj)
    assert obj.transfers == []
