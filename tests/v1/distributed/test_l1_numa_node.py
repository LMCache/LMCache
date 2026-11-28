# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``--l1-numa-node`` first-touch placement of the lazy L1."""

# Standard
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
import ctypes
import json
import os
import time

# Third Party
import pytest

# First Party
from lmcache.v1 import system_detection
from lmcache.v1.distributed import config as config_module
from lmcache.v1.distributed.config import L1MemoryManagerConfig, parse_args
from lmcache.v1.memory_allocators import lazy_memory_allocator
from lmcache.v1.memory_allocators.lazy_memory_allocator import LazyMemoryAllocator
from lmcache.v1.system_detection import NUMAMapping, get_numa_node_cpus

_CHUNK = LazyMemoryAllocator.PIN_CHUNK_SIZE
_PAGE = os.sysconf("SC_PAGE_SIZE")
_LIBC = ctypes.CDLL(None, use_errno=True)


@dataclass(frozen=True)
class _Event:
    """A memset ("touch") or pin of ``[ptr, ptr + size)`` seen by the test."""

    kind: str
    ptr: int
    size: int
    cpus: frozenset[int]
    resident_pages: int


def _resident_pages(ptr: int, size: int) -> int:
    """Return how many pages of ``[ptr, ptr + size)`` are resident (mincore),
    or -1 for a range that is not page-aligned."""
    if ptr % _PAGE or size % _PAGE:
        return -1
    vec = (ctypes.c_ubyte * (size // _PAGE))()
    assert _LIBC.mincore(ctypes.c_void_p(ptr), ctypes.c_size_t(size), vec) == 0
    return sum(v & 1 for v in vec)


def _usable_nodes() -> list[int]:
    """Return the NUMA nodes this process may run on and allocate from."""
    nodes = []
    for path in Path("/sys/devices/system/node").glob("node[0-9]*"):
        try:
            get_numa_node_cpus(int(path.name[4:]))
        except ValueError:
            continue
        nodes.append(int(path.name[4:]))
    return sorted(nodes)


def _pages_on_node(ptr: int, node: int) -> int:
    """Count resident pages on ``node`` in the mapping that contains ``ptr``."""
    with open("/proc/self/maps") as f:
        for line in f:
            start, end = (int(x, 16) for x in line.split()[0].split("-"))
            if start <= ptr < end:
                break
        else:
            raise AssertionError(f"no mapping contains {ptr:#x}")
    with open("/proc/self/numa_maps") as f:
        for line in f:
            fields = line.split()
            if int(fields[0], 16) == start:
                counts = dict(kv.split("=") for kv in fields if kv.startswith("N"))
                return int(counts.get(f"N{node}", 0))
    raise AssertionError(f"no numa_maps entry for {start:#x}")


def _wait_for_pins(events: list[_Event], count: int) -> list[_Event]:
    """Wait until ``count`` chunks are pinned and return a snapshot of events."""
    deadline = time.monotonic() + 30
    while sum(e.kind == "pin" for e in events) < count:
        assert time.monotonic() < deadline, "lazy expansion did not finish"
        time.sleep(0.01)
    return list(events)


class _FakePinSpec:
    """Pin backend that records every pin instead of registering memory."""

    is_pin_supported = True

    def __init__(self, events: list[_Event]) -> None:
        self.events = events

    def pin_memory(self, ptr: int, size: int, flags: int = 0) -> bool:
        self.events.append(
            _Event(
                "pin",
                ptr,
                size,
                frozenset(os.sched_getaffinity(0)),
                _resident_pages(ptr, size),
            )
        )
        return True

    def unpin_memory(self, ptr: int) -> bool:
        return True


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[_Event]:
    """Record pins and memsets, with the calling thread's affinity, in order."""
    log: list[_Event] = []
    spec = _FakePinSpec(log)
    monkeypatch.setattr(lazy_memory_allocator, "current_device_spec", spec)
    monkeypatch.setattr(config_module, "current_device_spec", spec)
    real_memset = ctypes.memset

    def memset(ptr: int, value: int, size: int) -> int:
        log.append(
            _Event(
                "touch",
                ptr,
                size,
                frozenset(os.sched_getaffinity(0)),
                _resident_pages(ptr, size),
            )
        )
        return real_memset(ptr, value, size)

    monkeypatch.setattr(ctypes, "memset", memset)
    return log


@pytest.fixture
def fake_sysfs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Callable[..., None]]:
    """Yield a setter that adds a node with a ``cpulist`` to a fake sysfs.

    Added nodes are listed in the fake ``Mems_allowed_list`` unless
    ``memory_allowed`` is False.
    """
    monkeypatch.setattr(system_detection, "_NUMA_NODE_SYSFS", str(tmp_path))
    status = tmp_path / "status"
    monkeypatch.setattr(system_detection, "_PROC_SELF_STATUS", str(status))
    mems: list[str] = []

    def add_node(node: int, cpulist: str, memory_allowed: bool = True) -> None:
        (tmp_path / f"node{node}").mkdir()
        (tmp_path / f"node{node}" / "cpulist").write_text(f"{cpulist}\n")
        if memory_allowed:
            mems.append(str(node))
        status.write_text(f"Name:\tpytest\nMems_allowed_list:\t{','.join(mems)}\n")

    yield add_node


# ---- get_numa_node_cpus ------------------------------------------------------


def test_node_cpus_parse_ranges_and_intersect_affinity(
    fake_sysfs: Callable[..., None],
) -> None:
    first = min(os.sched_getaffinity(0))
    fake_sysfs(0, f"{first}-{first},{first + 100000}-{first + 100002}")
    assert get_numa_node_cpus(0) == {first}


def test_node_cpus_reject_missing_node(fake_sysfs: Callable[..., None]) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        get_numa_node_cpus(7)


def test_node_cpus_reject_node_outside_cpu_affinity(
    fake_sysfs: Callable[..., None],
) -> None:
    fake_sysfs(1, "100000-100003")
    with pytest.raises(ValueError, match="CPU affinity"):
        get_numa_node_cpus(1)


def test_node_cpus_reject_node_outside_memory_cpuset(
    fake_sysfs: Callable[..., None],
) -> None:
    fake_sysfs(0, str(min(os.sched_getaffinity(0))))
    fake_sysfs(1, str(min(os.sched_getaffinity(0))), memory_allowed=False)
    assert get_numa_node_cpus(0)
    with pytest.raises(ValueError, match="may not allocate memory"):
        get_numa_node_cpus(1)


# ---- configuration -----------------------------------------------------------


def test_cli_and_json_set_numa_node(events: list[_Event]) -> None:
    legacy = parse_args(["--l1-size-gb", "1", "--eviction-policy", "LRU"])
    assert legacy.l1_manager_config.memory_config.numa_node is None

    legacy = parse_args(
        ["--l1-size-gb", "1", "--eviction-policy", "LRU", "--l1-numa-node", "1"]
    )
    assert legacy.l1_manager_config.memory_config.numa_node == 1

    dram = {"type": "DRAM", "tag": "_default", "size_gb": 1, "numa_node": 1}
    tagged = parse_args(["--eviction-policy", "LRU", "--l1-manager", json.dumps(dram)])
    assert tagged.l1_manager_config.memory_config.numa_node == 1


def test_cli_flag_rejected_with_tagged_l1(events: list[_Event]) -> None:
    dram = {"type": "DRAM", "tag": "_default", "size_gb": 1}
    with pytest.raises(ValueError, match="legacy L1 flags"):
        parse_args(
            [
                "--eviction-policy",
                "LRU",
                "--l1-manager",
                json.dumps(dram),
                "--l1-numa-node",
                "1",
            ]
        )


def test_cli_flag_rejected_with_gds_l1(events: list[_Event], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not apply to --gds-l1-path"):
        parse_args(
            [
                "--l1-size-gb",
                "1",
                "--eviction-policy",
                "LRU",
                "--gds-l1-path",
                str(tmp_path / "slab"),
                "--l1-numa-node",
                "1",
            ]
        )


@pytest.mark.parametrize("value", ["1", 1.0, True])
def test_json_rejects_non_integer_numa_node(
    events: list[_Event], value: object
) -> None:
    dram = {"type": "DRAM", "tag": "_default", "size_gb": 1, "numa_node": value}
    with pytest.raises(ValueError, match="numa_node must be an integer"):
        parse_args(["--eviction-policy", "LRU", "--l1-manager", json.dumps(dram)])


def test_numa_node_rejects_negative(events: list[_Event]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        L1MemoryManagerConfig(
            size_in_bytes=_CHUNK, use_lazy=True, shm_name="", numa_node=-1
        )


def test_numa_node_requires_lazy_allocation(events: list[_Event]) -> None:
    with pytest.raises(ValueError, match="requires lazy allocation"):
        L1MemoryManagerConfig(
            size_in_bytes=_CHUNK, use_lazy=False, shm_name="", numa_node=0
        )


# ---- allocator ---------------------------------------------------------------


def test_allocator_rejects_numa_mapping_with_numa_node(events: list[_Event]) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        LazyMemoryAllocator(
            _CHUNK,
            _CHUNK,
            numa_mapping=NUMAMapping(gpu_to_numa_mapping={0: 0}),
            numa_node=0,
        )


def test_each_chunk_is_first_touched_from_node_cpus_before_pinning(
    events: list[_Event], fake_sysfs: Callable[..., None]
) -> None:
    caller_affinity = frozenset(os.sched_getaffinity(0))
    node_cpu = min(caller_affinity)
    fake_sysfs(3, str(node_cpu))
    allocator = LazyMemoryAllocator(_CHUNK, 2 * _CHUNK, numa_node=3)
    try:
        # One chunk is pinned by the constructor, one by the expansion thread.
        log = _wait_for_pins(events, 2)
        touches = [e for e in log if e.kind == "touch"]
        pins = [e for e in log if e.kind == "pin"]
        assert [(e.ptr, e.size) for e in touches] == [(e.ptr, e.size) for e in pins]
        assert all(
            log.index(t) < log.index(p) for t, p in zip(touches, pins, strict=True)
        )
        # The mapping is fresh, so the first touch faults in every page itself.
        # Later chunks may already hold a few pages of a transparent huge page
        # that the previous touch faulted in across the chunk boundary.
        assert touches[0].resident_pages == 0
        assert all(e.cpus == {node_cpu} for e in touches)
        # Affinity is restored before pinning and for the constructing thread.
        assert all(e.cpus == caller_affinity for e in pins)
        assert os.sched_getaffinity(0) == caller_affinity
    finally:
        allocator.close()


def test_chunks_are_not_touched_without_numa_node(events: list[_Event]) -> None:
    allocator = LazyMemoryAllocator(_CHUNK, 2 * _CHUNK)
    try:
        log = _wait_for_pins(events, 2)
        assert not [e for e in log if e.kind == "touch"]
    finally:
        allocator.close()


@pytest.mark.skipif(len(_usable_nodes()) < 2, reason="needs two usable NUMA nodes")
@pytest.mark.parametrize("pick", [0, -1])
def test_pool_lands_on_requested_node(events: list[_Event], pick: int) -> None:
    # Both the first and the last node must receive the pool, which the
    # scheduler's choice of CPU alone cannot achieve.
    target = _usable_nodes()[pick]
    allocator = LazyMemoryAllocator(2 * _CHUNK, 2 * _CHUNK, numa_node=target)
    try:
        ptr = allocator.get_underlying_buffer().data_ptr()
        assert _pages_on_node(ptr, target) >= 0.99 * (2 * _CHUNK // _PAGE)
    finally:
        allocator.close()
