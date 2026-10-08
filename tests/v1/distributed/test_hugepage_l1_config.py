# SPDX-License-Identifier: Apache-2.0
"""Configuration contract for distributed HugeTLB L1 DRAM."""

# Standard
from types import SimpleNamespace
import argparse
import os
import sys

# Third Party
import pytest

# First Party
import lmcache.v1.distributed.config as config_module
from lmcache.v1.distributed.config import (
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
)
from lmcache.v1.multiprocess.config import add_mp_server_args


@pytest.fixture
def cuda_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        config_module,
        "current_device_spec",
        SimpleNamespace(device_type="cuda", is_pin_supported=True),
    )
    monkeypatch.setattr(config_module.sys, "platform", "linux")
    monkeypatch.setattr(
        config_module,
        "import_module",
        lambda name: SimpleNamespace(
            **{
                op: object()
                for op in (
                    "alloc_hugepage_pinned_ptr",
                    "free_hugepage_pinned_ptr",
                    "alloc_hugepage_pinned_numa_ptr",
                    "free_hugepage_pinned_numa_ptr",
                )
            }
        ),
    )


def _parse(options: list[str], multiprocess: bool = False) -> L1MemoryManagerConfig:
    parser = argparse.ArgumentParser()
    if multiprocess:
        add_mp_server_args(parser)
    add_storage_manager_args(parser)
    args = parser.parse_args(
        ["--l1-size-gb", "1", "--eviction-policy", "LRU", *options]
    )
    return parse_args_to_config(args).l1_manager_config.memory_config


@pytest.mark.parametrize("multiprocess", [False, True])
def test_hugepage_cli_defaults_and_boolean_forms(
    cuda_backend: None, multiprocess: bool
) -> None:
    default = _parse([], multiprocess)
    assert default.use_hugepages is False
    assert default.use_lazy is True
    assert default.shm_name == (
        "" if multiprocess else f"lmcache_l1_pool_{os.getpid()}"
    )

    enabled = _parse(["--l1-use-hugepages"], multiprocess)
    assert enabled.use_hugepages is True
    assert enabled.use_lazy is False
    assert enabled.shm_name == ""

    explicit = _parse(
        [
            "--l1-use-hugepages",
            "--no-l1-use-lazy",
            *(["--shm-name", ""] if multiprocess else []),
        ],
        multiprocess,
    )
    assert explicit.use_hugepages is True
    assert explicit.use_lazy is False
    assert explicit.shm_name == ""

    disabled = _parse(["--l1-use-hugepages", "--no-l1-use-hugepages"], multiprocess)
    assert disabled.use_hugepages is False


def test_hugepage_direct_config_defaults_and_explicit_eager(
    cuda_backend: None,
) -> None:
    config = L1MemoryManagerConfig(size_in_bytes=1 << 30, use_hugepages=True)
    assert config.use_lazy is False
    assert config.shm_name == ""
    explicit = L1MemoryManagerConfig(
        size_in_bytes=1 << 30, use_hugepages=True, use_lazy=False, shm_name=""
    )
    assert explicit.use_lazy is False
    assert explicit.shm_name == ""


@pytest.mark.parametrize("option", [["--l1-use-lazy"], ["--shm-name", "named"]])
@pytest.mark.parametrize("hugepage_first", [False, True])
def test_hugepage_cli_rejects_conflicts_in_both_orders(
    cuda_backend: None, option: list[str], hugepage_first: bool
) -> None:
    options = (
        ["--l1-use-hugepages", *option]
        if hugepage_first
        else [*option, "--l1-use-hugepages"]
    )
    with pytest.raises(ValueError, match=option[0]):
        _parse(options, multiprocess=True)


@pytest.mark.parametrize(
    "settings, conflict",
    [
        ({"use_lazy": True}, "--l1-use-lazy"),
        ({"shm_name": "named"}, "--shm-name"),
    ],
)
def test_hugepage_direct_config_rejects_conflicts(
    cuda_backend: None, settings: dict[str, object], conflict: str
) -> None:
    with pytest.raises(ValueError, match=conflict):
        L1MemoryManagerConfig(size_in_bytes=1 << 30, use_hugepages=True, **settings)


def test_hugepage_accepts_hybrid_but_rejects_pure_devdax_and_gds(
    cuda_backend: None,
) -> None:
    hybrid = L1MemoryManagerConfig(
        size_in_bytes=1 << 30,
        use_hugepages=True,
        devdax_path="/dev/dax0.0",
        devdax_size_in_bytes=1 << 30,
    )
    StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(hybrid),
        eviction_config=EvictionConfig("LRU"),
    )
    pure = L1MemoryManagerConfig(
        size_in_bytes=1 << 30, use_hugepages=True, devdax_path="/dev/dax0.0"
    )
    with pytest.raises(ValueError, match="pure --l1-devdax-path"):
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(pure),
            eviction_config=EvictionConfig("LRU"),
        )
    with pytest.raises(ValueError, match="--gds-l1-path"):
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=1 << 30, use_hugepages=True),
            gds_l1_config=GdsL1Config("/tmp/gds", 1 << 30),
        )


def test_hugepage_rejects_unsupported_backend_and_unsafe_sizes(
    cuda_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    for size in (0, -1, sys.maxsize):
        with pytest.raises(ValueError, match="size|--l1-size-gb"):
            L1MemoryManagerConfig(size_in_bytes=size, use_hugepages=True)
    monkeypatch.setattr(
        config_module,
        "current_device_spec",
        SimpleNamespace(device_type="cpu", is_pin_supported=False),
    )
    with pytest.raises(ValueError, match="CUDA native backend"):
        L1MemoryManagerConfig(size_in_bytes=1 << 30, use_hugepages=True)
    monkeypatch.setattr(
        config_module,
        "current_device_spec",
        SimpleNamespace(device_type="cuda", is_pin_supported=True),
    )
    monkeypatch.setattr(config_module.sys, "platform", "darwin")
    with pytest.raises(ValueError, match="Linux"):
        L1MemoryManagerConfig(size_in_bytes=1 << 30, use_hugepages=True)


def test_hugepage_rejects_missing_native_allocator(
    cuda_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_extension(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(config_module, "import_module", missing_extension)
    with pytest.raises(ValueError, match="lmcache.cuda_ops native extension"):
        L1MemoryManagerConfig(size_in_bytes=1 << 30, use_hugepages=True)


def test_server_rejects_shm_override_before_capacity_fallback(
    cuda_backend: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # First Party
    from lmcache.v1.mp_observability.config import DEFAULT_OBSERVABILITY_CONFIG
    from lmcache.v1.multiprocess.config import MPServerConfig
    import lmcache.v1.multiprocess.server as server

    storage = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=1 << 30, use_hugepages=True)
        ),
        eviction_config=EvictionConfig("LRU"),
    )

    def unexpected_capacity_check(path: str) -> None:
        raise AssertionError(f"capacity check reached for {path}")

    monkeypatch.setattr(server.shutil, "disk_usage", unexpected_capacity_check)
    with pytest.raises(ValueError, match="--shm-name"):
        server.run_cache_server(
            MPServerConfig(shm_name="named", supported_transfer_mode="engine_driven"),
            storage,
            DEFAULT_OBSERVABILITY_CONFIG,
        )
