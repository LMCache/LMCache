# SPDX-License-Identifier: Apache-2.0
"""Tests for the ROCm ``--l1-mlock`` option of the DRAM L1."""

# Standard
from types import SimpleNamespace
import json
import resource

# Third Party
import pytest

# First Party
from lmcache.v1.distributed import config as config_module
from lmcache.v1.distributed.config import L1MemoryManagerConfig, parse_args
from lmcache.v1.distributed.memory_manager import L1MemoryManager
from lmcache.v1.platform import current_device_spec
from tests.v1.distributed.utils import should_use_lazy_alloc

_SIZE = 64 << 20
_CAP_IPC_LOCK = 14


@pytest.fixture
def node(monkeypatch, tmp_path):
    """Pretend to run on ROCm; return a setter for the compaction sysctl."""
    monkeypatch.setattr(
        config_module,
        "current_device_spec",
        SimpleNamespace(
            backend_name="rocm", is_pin_supported=current_device_spec.is_pin_supported
        ),
    )
    sysctl = tmp_path / "compact_unevictable_allowed"
    monkeypatch.setattr(config_module, "_COMPACT_UNEVICTABLE_ALLOWED", str(sysctl))

    def set_sysctl(value: str) -> None:
        sysctl.write_text(f"{value}\n")

    set_sysctl("0")
    return set_sysctl


def _config(**kwargs) -> L1MemoryManagerConfig:
    return L1MemoryManagerConfig(
        size_in_bytes=_SIZE,
        use_lazy=should_use_lazy_alloc(),
        init_size_in_bytes=_SIZE,
        shm_name="",
        **kwargs,
    )


def _vm_flags(addr: int) -> list[str]:
    """Return the VmFlags of the mapping that contains ``addr``."""
    found = False
    with open("/proc/self/smaps") as f:
        for line in f:
            head = line.split()[0]
            if "-" in head and not head.endswith(":"):
                start, end = (int(x, 16) for x in head.split("-"))
                found = start <= addr < end
            elif found and head == "VmFlags:":
                return line.split()[1:]
    raise AssertionError(f"no mapping contains {addr:#x}")


def test_cli_and_json_enable_mlock(node):
    legacy = parse_args(["--l1-size-gb", "1", "--eviction-policy", "LRU"])
    assert legacy.l1_manager_config.memory_config.mlock is False

    legacy = parse_args(["--l1-size-gb", "1", "--eviction-policy", "LRU", "--l1-mlock"])
    assert legacy.l1_manager_config.memory_config.mlock is True

    dram = {"type": "DRAM", "tag": "_default", "size_gb": 1, "mlock": True}
    tagged = parse_args(["--eviction-policy", "LRU", "--l1-manager", json.dumps(dram)])
    assert tagged.l1_manager_config.memory_config.mlock is True


def test_mlock_rejected_when_compaction_may_move_locked_pages(node):
    node("1")
    with pytest.raises(RuntimeError, match="vm.compact_unevictable_allowed=0"):
        _config(mlock=True)


def test_mlock_rejected_off_rocm(node, monkeypatch):
    monkeypatch.setattr(
        config_module,
        "current_device_spec",
        SimpleNamespace(backend_name="cuda", is_pin_supported=True),
    )
    with pytest.raises(ValueError, match="only supported on ROCm"):
        _config(mlock=True)


def test_mlock_rejected_with_devdax(node):
    with pytest.raises(ValueError, match="l1-devdax-path"):
        L1MemoryManagerConfig(
            size_in_bytes=_SIZE,
            use_lazy=False,
            shm_name="",
            devdax_path="/dev/dax0.0",
            mlock=True,
        )


@pytest.mark.parametrize("mlock", [False, True])
def test_l1_buffer_locked_only_with_mlock(node, mlock):
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if mlock and soft != resource.RLIM_INFINITY and soft < _SIZE:
        pytest.skip("RLIMIT_MEMLOCK is smaller than the test L1")
    manager = L1MemoryManager(_config(mlock=mlock))
    try:
        assert ("lo" in _vm_flags(manager.get_l1_memory_desc().ptr)) is mlock
    finally:
        manager.close()


def _has_cap_ipc_lock() -> bool:
    with open("/proc/self/status") as f:
        cap_eff = next(line for line in f if line.startswith("CapEff:"))
    return bool(int(cap_eff.split()[1], 16) >> _CAP_IPC_LOCK & 1)


@pytest.mark.skipif(_has_cap_ipc_lock(), reason="CAP_IPC_LOCK ignores the limit")
def test_mlock_failure_stops_startup(node):
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (4096, hard))
    try:
        with pytest.raises(RuntimeError, match="memlock"):
            L1MemoryManager(_config(mlock=True))
    finally:
        resource.setrlimit(resource.RLIMIT_MEMLOCK, (soft, hard))
