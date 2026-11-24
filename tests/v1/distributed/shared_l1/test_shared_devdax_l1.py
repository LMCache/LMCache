# SPDX-License-Identifier: Apache-2.0
"""A shared Device-DAX L1 is an ordinary DevDaxL1Manager whose objects a memory
orchestrator owns; these tests run several such managers on one region."""

# Standard
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from itertools import count
from pathlib import Path
from typing import Any
from unittest.mock import Mock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
)
from lmcache.v1.distributed.error import L1Error, L1ReconfigureError
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.l1_manager import DevDaxL1Manager
from lmcache.v1.distributed.shared_l1 import visibility
from lmcache.v1.memory_orchestrator.api import (
    OrchestratorUnavailableError,
    ReadGrantResult,
    ReadRequest,
    ReadStatus,
    RegionContract,
    RegionFencedError,
    RegionUsage,
    RequestRejectedError,
    WireLayout,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.platform import current_device_spec
from tests.v1.distributed.shared_l1.utils import (
    CAPACITY,
    key,
    region_file,
    shared_config,
)
from tests.v1.memory_orchestrator.harness import Orchestrator

pytestmark = pytest.mark.no_shared_allocator
Factory = Callable[..., DevDaxL1Manager]
Usage = Callable[[], RegionUsage]
LAYOUT = MemoryLayoutDesc([torch.Size([2, 1000])], [torch.float16])
# Grant corruptions a server must refuse (LAYOUT's payload is 4000 bytes).
BAD_GRANTS: dict[str, Callable[[Any], Any]] = {
    "negative": lambda g: replace(g, handle=replace(g.handle, offset=-4096)),
    "misaligned": lambda g: replace(
        g, handle=replace(g.handle, offset=g.handle.offset + 1)
    ),
    "outside": lambda g: replace(g, handle=replace(g.handle, offset=CAPACITY)),
    "short": lambda g: replace(g, handle=replace(g.handle, length=2048)),
    "payload": lambda g: replace(g, payload_bytes=CAPACITY + 1),
    "layout": lambda g: replace(g, layout=WireLayout(((4, 500),), ("float16",))),
    "leases": lambda g: replace(g, leases=g.leases * 2),
}


def write(manager: DevDaxL1Manager, index: int = 1) -> L1OperationResult:
    return manager.reserve_write([key(index)], [False], LAYOUT)[key(index)]


def store(manager: DevDaxL1Manager, index: int = 1, value: int = 7) -> None:
    error, obj = write(manager, index)
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    obj.tensor.fill_(value)
    assert manager.finish_write([key(index)])[key(index)] is L1Error.SUCCESS


def patch_grants(monkeypatch: pytest.MonkeyPatch, rpc: str, fault: str) -> None:
    """Make ``MemoryOrchestratorClient.<rpc>`` return its grants with ``fault``."""
    real = getattr(MemoryOrchestratorClient, rpc)
    monkeypatch.setattr(
        MemoryOrchestratorClient,
        rpc,
        lambda client, entries: [BAD_GRANTS[fault](g) for g in real(client, entries)],
    )


@pytest.mark.parametrize("visibility_mode", ["coherent", "software_fenced"])
def test_peer_discovers_published_object_through_another_path(
    make_manager: Factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visibility_mode: str,
) -> None:
    flushed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        visibility, "cache_flush_range", lambda ptr, size: flushed.append((ptr, size))
    )
    monkeypatch.setattr(visibility, "cache_flush_supported", lambda: True)
    offset, fenced = 4096, visibility_mode == "software_fenced"
    device = region_file(tmp_path / "device", offset + CAPACITY)
    alias = tmp_path / "alias"
    alias.symlink_to(device)
    writer = make_manager("writer", path=device, offset_bytes=offset)
    reader = make_manager("reader", path=alias, offset_bytes=offset)
    error, obj = write(writer)
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    # No parent allocator: dropping the view never frees orchestrator bytes.
    assert obj.parent() is None
    obj.tensor.fill_(9)
    # Not readable anywhere until the writer publishes and commits it.
    assert reader.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert writer.finish_write([key(1)], tag="other")[key(1)] is L1Error.KEY_NOT_EXIST
    assert writer.finish_write([key(1)])[key(1)] is L1Error.SUCCESS
    # Software fencing writes the extent back after the copies, before the
    # commit, and invalidates it before a reader's copies; coherent never flushes.
    published = [(obj.data_ptr, obj.get_size())] if fenced else []
    assert flushed == published
    # The extent offset is region-relative; only the mapping adds the device offset.
    start = offset + obj.meta.address
    data = bytearray(device.read_bytes()[start : start + obj.get_size()])
    assert torch.all(torch.frombuffer(data, dtype=torch.float16) == 9)
    error, found = reader.reserve_read([key(1)])[key(1)]
    assert error is L1Error.SUCCESS and found is not None and found.tensor is not None
    acquired = [(found.data_ptr, found.get_size())] if fenced else []
    assert flushed == published + acquired
    assert torch.all(found.tensor == 9)
    assert found.get_l1_manager() == reader.l1_manager_id
    assert reader.unsafe_read([key(1)])[key(1)] == (L1Error.SUCCESS, found)
    assert reader.finish_read([key(1)])[key(1)] is L1Error.SUCCESS
    assert writer.owns_range(str(alias), offset + CAPACITY - 1, 1)
    assert not writer.owns_range(str(alias), 0, offset)


def test_concurrent_writers_and_overlapping_reads(
    make_manager: Factory, usage: Usage
) -> None:
    managers = [make_manager(str(i)) for i in range(8)]
    barrier = threading.Barrier(8)

    def reserve(manager: DevDaxL1Manager) -> L1OperationResult:
        barrier.wait(timeout=10)
        return write(manager)

    with ThreadPoolExecutor(8) as workers:
        results = list(workers.map(reserve, managers))
    assert [error for error, _ in results].count(L1Error.SUCCESS) == 1
    index = next(i for i, (error, _) in enumerate(results) if error is L1Error.SUCCESS)
    writer, reader = managers[index], managers[(index + 1) % 8]
    obj = results[index][1]
    assert obj is not None and obj.tensor is not None
    obj.tensor.fill_(7)
    writer.finish_write([key(1)])
    assert write(reader)[0] is L1Error.KEY_NOT_WRITABLE
    # A conflicting layout is an error, not capacity pressure.
    other = MemoryLayoutDesc([torch.Size([1000, 2])], [torch.float16])
    with pytest.raises(RequestRejectedError):
        reader.reserve_write([key(1)], [False], other)
    assert reader.get_staging_memory_usage() == 0
    with ThreadPoolExecutor(8) as workers:
        reads = list(workers.map(reader.reserve_read, [[key(1)]] * 8))
    assert all(ret[key(1)][0] is L1Error.SUCCESS for ret in reads)
    # Finishing one reader leaves the others protected.
    for _ in range(7):
        reader.finish_read([key(1)])
        assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    assert usage().read_leases == 1
    reader.finish_read([key(1)])
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    assert (usage().allocated_bytes, usage().read_leases) == (4096, 0)


@pytest.mark.parametrize("force", [False, True])
def test_delete_hides_the_object_everywhere_without_reclaiming(
    make_manager: Factory, usage: Usage, force: bool
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    leased = reader.reserve_read([key(1)])[key(1)][1]
    assert leased is not None and leased.tensor is not None
    writer.reserve_write([key(2)], [False], LAYOUT)
    assert writer.delete([key(1), key(2), key(3)], force=force) == {
        key(1): L1Error.SUCCESS,
        key(2): L1Error.KEY_IS_LOCKED,  # still being written
        key(3): L1Error.KEY_NOT_EXIST,
    }
    assert not writer.is_key_evictable(key(1))
    assert reader.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    # A lease taken before the delete keeps its bytes until it is released.
    assert reader.unsafe_read([key(1)])[key(1)] == (L1Error.SUCCESS, leased)
    assert torch.all(leased.tensor == 7)
    assert reader.finish_read([key(1)])[key(1)] is L1Error.SUCCESS
    # Nothing was reclaimed: storing the key again takes a new extent.
    assert usage().allocated_bytes == 2 * 4096
    store(reader, value=9)
    assert usage().allocated_bytes == 3 * 4096
    writer.finish_write_and_delete([key(2)])


def test_clear_deletes_only_what_this_server_wrote(
    make_manager: Factory, usage: Usage
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    store(writer, index=1)
    store(peer, index=2)
    writer.clear(force=True)
    assert peer.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert writer.reserve_read([key(2)])[key(2)][0] is L1Error.SUCCESS
    writer.finish_read([key(2)])
    assert (usage().valid, usage().allocated_bytes) == (1, 2 * 4096)


def test_delete_while_unreachable_is_reported_locked(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    offline = Mock(side_effect=OrchestratorUnavailableError("offline"))
    monkeypatch.setattr(MemoryOrchestratorClient, "delete_objects", offline)
    monkeypatch.setattr(MemoryOrchestratorClient, "delete_published", offline)
    assert writer.delete([key(1)]) == {key(1): L1Error.KEY_IS_LOCKED}
    writer.clear()
    monkeypatch.undo()
    assert reader.reserve_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    reader.finish_read([key(1)])


@pytest.mark.parametrize("failure", ["abort", "publish", "grant"])
def test_failed_write_is_writable_again_but_bytes_stay_consumed(
    make_manager: Factory, usage: Usage, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    with monkeypatch.context() as patch:
        if failure == "grant":
            patch_grants(patch, "reserve_write", "misaligned")
            assert write(writer)[0] is L1Error.KEY_NOT_WRITABLE
        elif failure == "publish":
            write(writer)
            failed = Mock(side_effect=RuntimeError)
            patch.setattr(visibility.Visibility, "publish", failed)
            with pytest.raises(RuntimeError):
                writer.finish_write([key(1)])
        else:
            write(writer)
            assert writer.finish_write_and_delete([key(1)]) == {key(1): L1Error.SUCCESS}
    assert writer.get_staging_memory_usage() == 0
    store(peer)
    assert usage().allocated_bytes == 2 * 4096


def test_region_facts_refused_calls_and_close(
    make_manager: Factory, region: Path, usage: Usage
) -> None:
    manager = make_manager("owner")
    remote = MemoryTags(MemorySegment.CXL, MemoryResidency.REMOTE)
    assert manager.get_memory_tags() == remote
    assert manager.get_l1_memory_desc() is None
    assert manager.memory_region_count() == 1
    assert manager.owns_range(str(region), 0, 1)
    assert not manager.owns_range(str(region), CAPACITY, 4096)
    assert manager.get_capacity_bytes() == CAPACITY
    assert manager.get_memory_usage() == (0, CAPACITY)
    status = manager.report_status()
    assert status["is_healthy"] and status["shared"]["state"] == "available"
    assert manager.get_object_state(key(1)) is None
    with pytest.raises(ValueError, match="temporary"):
        manager.reserve_write([key(1)], [True], LAYOUT)
    for call in (
        manager.get_devdax_arena_statuses,
        lambda: manager.add_devdax_device(str(region), 4096),
        lambda: manager.remove_devdax_device(str(region)),
    ):
        with pytest.raises(L1ReconfigureError) as raised:
            call()
        assert raised.value.status_code == 409
    # Refused before mutating any state: the manager still stores.
    store(manager)
    write(manager, 2)
    manager.reserve_read([key(1)], read_locks=2)
    held = manager.unsafe_read([key(1)])
    with pytest.raises(BufferError):
        manager.close()
    assert manager.reserve_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    del held
    manager.close()
    assert (usage().writing, usage().read_leases, usage().clients) == (0, 0, 1)


def test_finish_write_and_reserve_read_reports_only_what_succeeded(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager("owner")
    events: dict[EventType, dict] = {}
    monkeypatch.setattr(
        get_event_bus(), "publish", lambda e: events.update({e.event_type: e.metadata})
    )
    manager.reserve_write([key(1), key(2)], [False, False], LAYOUT)
    reserve_read = MemoryOrchestratorClient.reserve_read

    def lease_only_first(
        client: MemoryOrchestratorClient, entries: list[ReadRequest]
    ) -> list[ReadGrantResult]:
        # The second key commits, but its lease is refused.
        refused = ReadGrantResult(ReadStatus.BUSY_WRITING)
        return reserve_read(client, entries[:1]) + [refused]

    monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", lease_only_first)
    ret = manager.finish_write_and_reserve_read([key(1), key(2)])
    assert ret[key(1)][0] is L1Error.SUCCESS
    assert ret[key(2)] == (L1Error.KEY_NOT_EXIST, None)
    combined = events[EventType.L1_WRITE_FINISHED_AND_READ_RESERVED]
    finished = events[EventType.L1_WRITE_FINISHED]
    assert (combined["keys"], finished["keys"]) == ([key(1)], [key(2)])
    assert combined["shared"] and finished["shared"]
    manager.finish_read([key(1)])


@pytest.mark.parametrize("state", ["disabled", "unavailable"])
def test_disabled_or_unavailable_region_misses_reads_and_fails_writes(
    make_manager: Factory, orchestrator: Orchestrator, state: str
) -> None:
    manager = make_manager("owner", rpc_timeout_seconds=0.5)
    store(manager)
    assert manager.reserve_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    if state == "disabled":
        make_manager("owner")  # A newer incarnation retires the held lease.
    else:
        orchestrator.stop()
    assert write(manager, 2)[0] is L1Error.OUT_OF_MEMORY
    assert manager.reserve_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    if state == "disabled":
        assert manager.unsafe_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    assert not manager.memcheck()
    status = manager.report_status()
    assert status["shared"]["state"] == state and not status["is_healthy"]


@pytest.mark.parametrize(
    "change,error",
    [
        ({"reset_required": True}, RuntimeError),
        ({"capacity_bytes": CAPACITY * 2}, ValueError),
        ({"visibility_mode": "software_fenced", "alignment_bytes": 64}, ValueError),
        ({"visibility_mode": "unknown"}, ValueError),
        # Region epochs are positive, so the region never had epoch 0.
        ({"region_epoch": 0}, RegionFencedError),
    ],
)
def test_startup_contract_failure_cleans_up(
    make_manager: Factory, usage: Usage, change: dict[str, Any], error: type[Exception]
) -> None:
    describe, calls = MemoryOrchestratorClient.describe_region, count()

    def changed(client: MemoryOrchestratorClient) -> RegionContract:
        # register() describes the region again and must see the real contract.
        contract = describe(client)
        return replace(contract, **change) if next(calls) == 0 else contract

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MemoryOrchestratorClient, "describe_region", changed)
        with pytest.raises(error):
            make_manager("bad")
    assert usage().clients == 1


def test_unmappable_region_is_rejected_before_registering(
    make_manager: Factory, usage: Usage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="exceeds"):
        make_manager("bad", path=region_file(tmp_path / "small", CAPACITY // 2))
    assert usage().clients == 1
    monkeypatch.setattr(current_device_spec, "pin_memory", lambda *args: False)
    with pytest.raises(RuntimeError, match="GPU-registered"):
        make_manager("bad")
    assert usage().clients == 1


def test_ambiguous_commit_is_reported_failed(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = make_manager("writer")
    write(writer)
    lost = Mock(side_effect=OrchestratorUnavailableError)
    monkeypatch.setattr(MemoryOrchestratorClient, "finish_write", lost)
    assert writer.finish_write([key(1)]) == {key(1): L1Error.KEY_IN_WRONG_STATE}
    # The extent is never reused here: the grant is gone locally.
    assert writer.get_staging_memory_usage() == 0
    assert writer.finish_write([key(1)]) == {key(1): L1Error.KEY_NOT_EXIST}


@pytest.mark.parametrize("fault", ["acquire", *BAD_GRANTS])
def test_failed_read_releases_only_the_new_lease(
    make_manager: Factory, usage: Usage, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    original = reader.reserve_read([key(1)])[key(1)]
    if fault == "acquire":
        failed = Mock(side_effect=RuntimeError)
        monkeypatch.setattr(visibility.Visibility, "acquire", failed)
        with pytest.raises(RuntimeError):
            reader.reserve_read([key(1)], read_locks=2)
    else:
        patch_grants(monkeypatch, "reserve_read", fault)
        assert reader.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert reader.unsafe_read([key(1)])[key(1)] == original
    assert usage().read_leases == 1
    reader.finish_read([key(1)])
    assert usage().read_leases == 0


def test_private_devdax_l1_never_contacts_an_orchestrator(
    region: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contact = Mock(side_effect=AssertionError("private L1 contacted an orchestrator"))
    monkeypatch.setattr(MemoryOrchestratorClient, "__init__", contact)
    config = replace(shared_config(region, "unused", "a"), shared=None)
    with closing(DevDaxL1Manager(config)) as private:
        store(private)
        assert private.get_memory_tags().residency is MemoryResidency.LOCAL
        assert private.delete([key(1)]) == {key(1): L1Error.SUCCESS}


def test_stalled_orchestrator_call_does_not_hold_the_manager_lock(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager("owner")
    store(manager)
    entered, release = threading.Event(), threading.Event()
    reserve_read = MemoryOrchestratorClient.reserve_read

    def stalled(
        client: MemoryOrchestratorClient, entries: list[ReadRequest]
    ) -> list[ReadGrantResult]:
        entered.set()
        assert release.wait(timeout=10)
        return reserve_read(client, entries)

    monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", stalled)
    with ThreadPoolExecutor(1) as worker:
        lookup = worker.submit(manager.reserve_read, [key(1)])
        assert entered.wait(timeout=10)
        # Operations that take the manager lock complete during the RPC.
        manager.register_listener(Mock())
        assert manager.report_status()["shared"]["state"] == "available"
        assert manager.get_staging_memory_usage() == 0
        release.set()
        assert lookup.result(timeout=10)[key(1)][0] is L1Error.SUCCESS
    manager.finish_read([key(1)])
