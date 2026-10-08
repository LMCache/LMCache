# SPDX-License-Identifier: Apache-2.0
"""Opt-in hardware checks for the distributed HugeTLB L1 arena.

Run with ``LMCACHE_TEST_HUGEPAGES=1 pytest -xvs`` on an isolated Linux CUDA
runner with a fixed 2 MiB HugeTLB pool. Skips never qualify as phase 3 evidence.
"""

# Standard
from pathlib import Path
from multiprocessing.connection import Connection
import ctypes
import gc
import multiprocessing
import os
import re
import select
import socket
import subprocess
import sys
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l2_adapters.nixl_store_l2_adapter import (
    NixlStoreL2Adapter,
    NixlStoreL2AdapterConfig,
)
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.distributed.transfer_channel.impl.nixl_impl import (
    NixlTransferChannelContext,
)
from lmcache.v1.platform import consume_fd

PAGE_SIZE = 2 * 1024 * 1024
POOL = Path("/sys/kernel/mm/hugepages/hugepages-2048kB")

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.integration,
    pytest.mark.no_shared_allocator,
]


@pytest.fixture(scope="module", autouse=True)
def require_hugepage_runner() -> None:
    """Require an explicit hardware run with CUDA and a fixed HugeTLB pool."""
    if os.environ.get("LMCACHE_TEST_HUGEPAGES") != "1":
        pytest.skip("set LMCACHE_TEST_HUGEPAGES=1 on a provisioned runner")
    assert sys.platform == "linux"
    assert torch.cuda.is_available()
    assert int((POOL / "nr_overcommit_hugepages").read_text()) == 0
    assert int((POOL / "nr_hugepages").read_text()) >= 512


def _free_pages() -> int:
    """Read the free count of the host's 2 MiB HugeTLB pool."""
    return int((POOL / "free_hugepages").read_text())


def _assert_pool_reclaimed(baseline: int) -> None:
    """Allow asynchronous deregistration a short window to release pages."""
    deadline = time.monotonic() + 5
    while _free_pages() != baseline and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _free_pages() == baseline


def _mapping(ptr: int) -> dict[str, str]:
    """Read the smaps entry containing a public L1 arena pointer."""
    entry: dict[str, str] = {}
    found = False
    for line in Path(f"/proc/{os.getpid()}/smaps").read_text().splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"[0-9a-f]+-[0-9a-f]+", fields[0]):
            start, end = fields[0].split("-", 1)
            if found:
                break
            found = int(start, 16) <= ptr < int(end, 16)
            if found:
                entry["range"] = fields[0]
            continue
        if found and ":" in line:
            key, value = line.split(":", 1)
            entry[key] = value.strip()
    assert entry, f"No smaps mapping contains L1 pointer {ptr:#x}"
    return entry


def _assert_hugetlb(ptr: int, mapped_size: int) -> None:
    """Verify address-specific HugeTLB flags and fully charged page bytes."""
    mapping = _mapping(ptr)
    assert "ht" in mapping["VmFlags"].split(), mapping
    charged_kb = sum(
        int(mapping[key].split()[0]) for key in ("Private_Hugetlb", "Shared_Hugetlb")
    )
    assert charged_kb * 1024 == mapped_size, mapping
    print(
        f"L1 mapping: {mapping['range']} {mapping['VmFlags']}; "
        f"HugeTLB charged={charged_kb} kB"
    )


def _manager(size: int) -> L1MemoryManager:
    """Create a public manager with real eager anonymous HugeTLB backing."""
    return L1MemoryManager(
        L1MemoryManagerConfig(size_in_bytes=size, use_hugepages=True)
    )


def _url_pair() -> tuple[str, str]:
    """Find distinct loopback ports for two NIXL transfer servers."""
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return (
            f"127.0.0.1:{first.getsockname()[1]}",
            f"127.0.0.1:{second.getsockname()[1]}",
        )


def _wait_event(fd: int) -> None:
    """Wait for an L2 task to signal completion through its public event fd."""
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    assert poller.poll(10_000), "L2 transfer did not finish within 10 seconds"
    consume_fd(fd)


def _nixl_source_process(pipe: Connection, url: str) -> None:
    """Serve one HugeTLB-backed NIXL object from a spawned process."""
    manager = _manager(PAGE_SIZE)
    context = None
    objects = []
    try:
        layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
        error, objects = manager.allocate(layout, 1)
        assert error == L1Error.SUCCESS
        objects[0].raw_tensor.copy_(
            torch.arange(4096, dtype=torch.int32).remainder(251).to(torch.uint8)
        )
        context = NixlTransferChannelContext(manager.get_l1_memory_desc(), url, url)
        pipe.send("ready")
        assert pipe.recv() == "close"
    finally:
        if context is not None:
            context.close()
        manager.free(objects)
        manager.close()
        pipe.close()


def _touch(ptr: int, size: int) -> None:
    """Fault every 2 MiB page of the rounded native mapping."""
    for offset in range(0, size, PAGE_SIZE):
        ctypes.c_uint8.from_address(ptr + offset).value = 0x5A


def test_one_gib_arena_returns_all_pages() -> None:
    """A touched 1 GiB arena uses exactly 512 HugeTLB pages until close."""
    baseline = _free_pages()
    manager = _manager(1 << 30)
    try:
        desc = manager.get_l1_memory_desc()
        assert desc.size == 1 << 30
        _touch(desc.ptr, desc.size)
        _assert_hugetlb(desc.ptr, desc.size)
        print(f"2 MiB pool: before={baseline}, during={_free_pages()}")
        assert baseline - _free_pages() == 512
    finally:
        manager.close()
    print(f"2 MiB pool: after={_free_pages()}")
    _assert_pool_reclaimed(baseline)


def test_unaligned_arena_repeats_without_leak() -> None:
    """One extra 4 KiB needs two pages and 100 lifetimes leave no pool debt."""
    logical_size = PAGE_SIZE + 4096
    baseline = _free_pages()
    for iteration in range(100):
        manager = _manager(logical_size)
        try:
            desc = manager.get_l1_memory_desc()
            assert desc.size == logical_size
            _touch(desc.ptr, 2 * PAGE_SIZE)
            if iteration == 0:
                _assert_hugetlb(desc.ptr, 2 * PAGE_SIZE)
                print(f"2 MiB pool: before={baseline}, during={_free_pages()}")
            assert baseline - _free_pages() == 2
        finally:
            manager.close()
        _assert_pool_reclaimed(baseline)
    print(f"2 MiB pool: after 100 cycles={_free_pages()}")


@pytest.mark.parametrize(
    "shape,dtype",
    [((4096,), torch.uint8), ((2048,), torch.float16), ((1024,), torch.float32)],
)
def test_gpu_roundtrip_and_reuse(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    """One thousand GPU store/retrieve cycles preserve bytes across layouts."""
    manager = _manager(2 * PAGE_SIZE)
    layout = MemoryLayoutDesc(shapes=[torch.Size(shape)], dtypes=[dtype])
    try:
        for iteration in range(1000):
            error, objects = manager.allocate(layout, count=1)
            assert error == L1Error.SUCCESS
            obj = objects[0]
            try:
                source = torch.full(shape, iteration % 251, dtype=dtype, device="cuda")
                source_bytes = source.view(torch.uint8)
                obj.raw_tensor.copy_(source_bytes, non_blocking=True)
                restored = torch.empty_like(source_bytes)
                restored.copy_(obj.raw_tensor, non_blocking=True)
                torch.cuda.synchronize()
                assert torch.equal(source_bytes, restored)
            finally:
                assert manager.free(objects) == L1Error.SUCCESS
    finally:
        manager.close()


def test_two_nixl_contexts_transfer_hugepage_objects() -> None:
    """Two NIXL servers read byte-identical objects from HugeTLB L1 arenas."""
    pytest.importorskip("nixl._api")
    baseline = _free_pages()
    source_manager = _manager(PAGE_SIZE)
    destination_manager = _manager(PAGE_SIZE)
    contexts: list[NixlTransferChannelContext] = []
    layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
    source_objects = []
    destination_objects = []
    try:
        source_error, source_objects = source_manager.allocate(layout, 1)
        destination_error, destination_objects = destination_manager.allocate(layout, 1)
        assert source_error == destination_error == L1Error.SUCCESS
        expected = torch.arange(4096, dtype=torch.int32).remainder(251).to(torch.uint8)
        source_objects[0].raw_tensor.copy_(expected)
        destination_objects[0].raw_tensor.zero_()
        source_desc = source_manager.get_l1_memory_desc()
        destination_desc = destination_manager.get_l1_memory_desc()
        _assert_hugetlb(source_desc.ptr, PAGE_SIZE)
        _assert_hugetlb(destination_desc.ptr, PAGE_SIZE)
        source_url, destination_url = _url_pair()
        source = NixlTransferChannelContext(source_desc, source_url, source_url)
        contexts.append(source)
        destination = NixlTransferChannelContext(
            destination_desc, destination_url, destination_url
        )
        contexts.append(destination)
        client = destination.get_transfer_channel_client(source.advertise_url)
        local = destination.get_transfer_channel_address([(0, 4096)])
        remote = source.get_transfer_channel_address([(0, 4096)])
        task_id = client.submit_read(local, remote)
        deadline = time.monotonic() + 30
        result = client.query_read_status(task_id)
        while not result.is_finished() and time.monotonic() < deadline:
            time.sleep(0.01)
            result = client.query_read_status(task_id)
        assert result.is_finished()
        assert result.succeeded_mask == [True]
        assert torch.equal(destination_objects[0].raw_tensor, expected)
    finally:
        for context in reversed(contexts):
            context.close()
        source_manager.free(source_objects)
        destination_manager.free(destination_objects)
        source_manager.close()
        destination_manager.close()
    _assert_pool_reclaimed(baseline)


def test_two_process_nixl_transfer_hugepage_objects() -> None:
    """A separate NIXL server process transfers a HugeTLB-backed object."""
    pytest.importorskip("nixl._api")
    baseline = _free_pages()
    source_url, destination_url = _url_pair()
    spawn = multiprocessing.get_context("spawn")
    parent_pipe, child_pipe = spawn.Pipe()
    source_process = spawn.Process(
        target=_nixl_source_process, args=(child_pipe, source_url)
    )
    source_process.start()
    child_pipe.close()
    manager = _manager(PAGE_SIZE)
    context = None
    client = None
    objects = []
    try:
        assert parent_pipe.poll(30), "source server did not start"
        assert parent_pipe.recv() == "ready"
        layout = MemoryLayoutDesc(shapes=[torch.Size([4096])], dtypes=[torch.uint8])
        error, objects = manager.allocate(layout, 1)
        assert error == L1Error.SUCCESS
        context = NixlTransferChannelContext(
            manager.get_l1_memory_desc(), destination_url, destination_url
        )
        client = context.get_transfer_channel_client(source_url)
        local = context.get_transfer_channel_address([(0, 4096)])
        remote = [TransferChannelAddress(offset=0, size=4096)]
        task_id = client.submit_read(local, remote)
        deadline = time.monotonic() + 30
        result = client.query_read_status(task_id)
        while not result.is_finished() and time.monotonic() < deadline:
            time.sleep(0.01)
            result = client.query_read_status(task_id)
        assert result.is_finished()
        assert result.succeeded_mask == [True]
        expected = torch.arange(4096, dtype=torch.int32).remainder(251).to(torch.uint8)
        assert torch.equal(objects[0].raw_tensor, expected)
        parent_pipe.send("close")
        source_process.join(timeout=30)
        assert source_process.exitcode == 0
    finally:
        if source_process.is_alive():
            source_process.terminate()
            source_process.join(timeout=10)
        parent_pipe.close()
        if context is not None:
            context.close()
        client = None
        context = None
        gc.collect()
        manager.free(objects)
        manager.close()
    _assert_pool_reclaimed(baseline)


def test_nixl_l2_reload_after_l1_eviction(tmp_path: Path) -> None:
    """One thousand NIXL L2 reloads preserve bytes after L1 eviction."""
    pytest.importorskip("nixl._api")
    baseline = _free_pages()
    manager = _manager(PAGE_SIZE)
    adapter = None
    objects = []
    try:
        desc = manager.get_l1_memory_desc()
        _assert_hugetlb(desc.ptr, PAGE_SIZE)
        adapter = NixlStoreL2Adapter(
            NixlStoreL2AdapterConfig(
                backend="POSIX",
                backend_params={"file_path": str(tmp_path), "use_direct_io": "false"},
                pool_size=1000,
            ),
            desc,
        )
        layouts = [
            ((4096,), torch.uint8),
            ((2048,), torch.float16),
            ((1024,), torch.float32),
        ]
        for iteration in range(1000):
            shape, dtype = layouts[iteration % len(layouts)]
            layout = MemoryLayoutDesc(shapes=[torch.Size(shape)], dtypes=[dtype])
            key = ObjectKey(
                chunk_hash=ObjectKey.IntHash2Bytes(iteration + 1),
                model_name="hugepage",
                kv_rank=0,
            )
            error, objects = manager.allocate(layout, 1)
            assert error == L1Error.SUCCESS
            source = torch.full(shape, iteration % 251, dtype=dtype, device="cuda")
            source_bytes = source.view(torch.uint8)
            objects[0].raw_tensor.copy_(source_bytes, non_blocking=True)
            torch.cuda.synchronize()
            store_id = adapter.submit_store_task([key], objects)
            _wait_event(adapter.get_store_event_fd())
            assert adapter.pop_completed_store_tasks()[store_id].is_successful()
            assert manager.free(objects) == L1Error.SUCCESS
            objects = []
            error, objects = manager.allocate(layout, 1)
            assert error == L1Error.SUCCESS
            objects[0].raw_tensor.zero_()
            load_id = adapter.submit_load_task([key], objects)
            _wait_event(adapter.get_load_event_fd())
            assert adapter.query_load_result(load_id).test(0)
            restored = torch.empty_like(source_bytes)
            restored.copy_(objects[0].raw_tensor, non_blocking=True)
            torch.cuda.synchronize()
            assert torch.equal(restored, source_bytes)
            assert manager.free(objects) == L1Error.SUCCESS
            objects = []
    finally:
        if adapter is not None:
            adapter.close()
        manager.free(objects)
        manager.close()
    _assert_pool_reclaimed(baseline)


@pytest.mark.parametrize(
    "scenario,expected_error",
    [
        ("exhausted_pool", "mmap failed"),
        ("registration", "cudaHostRegister failed"),
    ],
)
def test_native_failure_reclaims_mapping(scenario: str, expected_error: str) -> None:
    """Fatal mapping and registration failures leave the fixed pool intact."""
    script = """
from pathlib import Path
import sys
from lmcache import device_ops

pool = Path('/sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages')
baseline = int(pool.read_text())
page_size = 2 * 1024 * 1024
scenario = sys.argv[1]
size = (baseline + 1) * page_size if scenario == 'exhausted_pool' else page_size
flags = 0 if scenario == 'exhausted_pool' else 0xffffffff
try:
    ptr = device_ops.alloc_hugepage_pinned_ptr(size, flags)
except RuntimeError as error:
    print(str(error))
    assert int(pool.read_text()) == baseline
else:
    device_ops.free_hugepage_pinned_ptr(ptr, size)
    raise AssertionError('native allocation unexpectedly succeeded')
"""
    baseline = _free_pages()
    result = subprocess.run(
        [sys.executable, "-c", script, scenario],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected_error in result.stdout
    _assert_pool_reclaimed(baseline)
