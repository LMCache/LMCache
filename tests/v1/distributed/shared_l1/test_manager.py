# SPDX-License-Identifier: Apache-2.0
"""Shared-only ownership/failure contracts; basic lifecycle uses test_l1_interface."""

# Standard
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest.mock import Mock
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1OperationResult
from lmcache.v1.distributed.shared_l1.manager import SharedL1Manager
from lmcache.v1.distributed.shared_l1.visibility import Visibility
from lmcache.v1.memory_orchestrator.api import (
    OrchestratorError,
    OrchestratorUnavailableError,
    ReadGrantResult,
    ReadRequest,
    RegionContract,
    RegionFencedError,
    RegionUsage,
    RequestRejectedError,
    WireLayout,
    WriteGrantResult,
    WriteRequest,
)
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from tests.v1.distributed.shared_l1.utils import CAPACITY, key

pytestmark = pytest.mark.no_shared_allocator
Factory = Callable[[str], SharedL1Manager]
Usage = Callable[[], RegionUsage]
LAYOUT = MemoryLayoutDesc([torch.Size([2, 1000])], [torch.float16])


def store(manager: SharedL1Manager) -> None:
    error, obj = manager.reserve_write([key(1)], [False], LAYOUT)[key(1)]
    assert error is L1Error.SUCCESS and obj is not None and obj.tensor is not None
    obj.tensor.fill_(7)
    assert manager.finish_write([key(1)])[key(1)] is L1Error.SUCCESS


def test_concurrent_publication_and_overlapping_reads(
    make_manager: Factory, usage: Usage
) -> None:
    managers = [make_manager(str(i)) for i in range(8)]
    barrier = threading.Barrier(8)

    def reserve(manager: SharedL1Manager) -> L1OperationResult:
        barrier.wait(timeout=10)
        return manager.reserve_write([key(1)], [False], LAYOUT)[key(1)]

    with ThreadPoolExecutor(8) as workers:
        results = list(workers.map(reserve, managers))
    assert [error for error, _ in results].count(L1Error.SUCCESS) == 1
    index = next(i for i, (error, _) in enumerate(results) if error is L1Error.SUCCESS)
    writer, reader = managers[index], managers[(index + 1) % 8]
    obj = results[index][1]
    assert obj is not None and obj.tensor is not None
    obj.tensor.fill_(7)
    writer.finish_write([key(1)])
    writer.clear(force=True)
    assert writer.delete([key(1)], force=True)[key(1)] is L1Error.KEY_IS_LOCKED
    assert not writer.is_key_evictable(key(1))
    assert (
        reader.reserve_write([key(1)], [False], LAYOUT)[key(1)][0]
        is L1Error.KEY_NOT_WRITABLE
    )
    with ThreadPoolExecutor(8) as workers:
        reads = list(
            workers.map(lambda _: reader.reserve_read([key(1)])[key(1)], range(8))
        )
    assert all(error is L1Error.SUCCESS for error, _ in reads)
    obj = reads[0][1]
    assert obj is not None and obj.tensor is not None
    assert torch.all(obj.tensor == 7)
    for _ in range(7):
        reader.finish_read([key(1)])
        assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.SUCCESS
    reader.finish_read([key(1)])
    assert reader.unsafe_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
    assert (usage().allocated_bytes, usage().read_leases) == (4096, 0)


@pytest.mark.parametrize("state", ["write", "read"])
def test_close_retires_owned_state(
    make_manager: Factory, usage: Usage, state: str
) -> None:
    manager = make_manager("owner")
    assert (
        manager.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is L1Error.SUCCESS
    )
    if state == "read":
        manager.finish_write([key(1)])
        manager.reserve_read([key(1)], read_locks=2)
    manager.close()
    assert (usage().writing, usage().read_leases, usage().clients) == (0, 0, 1)


@pytest.mark.parametrize(
    "failure", [OrchestratorUnavailableError("offline"), RegionFencedError("stale")]
)
def test_rpc_failure_stops_serving(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch, failure: OrchestratorError
) -> None:
    manager = make_manager("owner")
    store(manager)
    with monkeypatch.context() as patch:
        for method in ("reserve_write", "reserve_read"):
            patch.setattr(MemoryOrchestratorClient, method, Mock(side_effect=failure))
        assert (
            manager.reserve_write([key(2)], [False], LAYOUT)[key(2)][0]
            is L1Error.OUT_OF_MEMORY
        )
        assert manager.reserve_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST
        assert not manager.memcheck()
    if isinstance(failure, RegionFencedError):
        assert manager.reserve_read([key(1)])[key(1)][0] is L1Error.KEY_NOT_EXIST


@pytest.mark.parametrize(
    "change,error",
    [
        ({"reset_required": True}, RuntimeError),
        ({"capacity_bytes": CAPACITY * 2}, ValueError),
        ({"visibility_mode": "software_fenced", "alignment_bytes": 64}, ValueError),
        ({"region_epoch": 0}, RegionFencedError),
    ],
)
def test_startup_contract_failure_cleans_up(
    make_manager: Factory,
    usage: Usage,
    monkeypatch: pytest.MonkeyPatch,
    change: dict[str, object],
    error: type[Exception],
) -> None:
    describe = MemoryOrchestratorClient.describe_region
    calls = 0

    def changed(client: MemoryOrchestratorClient) -> RegionContract:
        nonlocal calls
        contract = describe(client)
        calls += 1
        update = (
            {"region_epoch": contract.region_epoch ^ 1}
            if "region_epoch" in change
            else change
        )
        return replace(contract, **update) if calls == 1 else contract

    monkeypatch.setattr(MemoryOrchestratorClient, "describe_region", changed)
    with pytest.raises(error):
        make_manager("bad")
    assert usage().clients == 1


def test_conflicting_layout_is_not_overflow(make_manager: Factory) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    with pytest.raises(RequestRejectedError):
        reader.reserve_write(
            [key(1)],
            [False],
            MemoryLayoutDesc([torch.Size([1000, 2])], [torch.float16]),
        )
    assert reader.get_staging_memory_usage() == 0


def test_failed_publish_aborts_the_write(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    writer.reserve_write([key(1)], [False], LAYOUT)
    monkeypatch.setattr(
        Visibility, "publish", Mock(side_effect=RuntimeError("publish"))
    )
    with pytest.raises(RuntimeError, match="publish"):
        writer.finish_write([key(1)])
    assert writer.get_staging_memory_usage() == 0
    assert peer.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is L1Error.SUCCESS


def test_failed_acquire_preserves_existing_lease(
    make_manager: Factory, usage: Usage, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    original = reader.reserve_read([key(1)])[key(1)]
    monkeypatch.setattr(
        Visibility, "acquire", Mock(side_effect=RuntimeError("acquire"))
    )
    with pytest.raises(RuntimeError, match="acquire"):
        reader.reserve_read([key(1)], read_locks=2)
    assert reader.unsafe_read([key(1)])[key(1)] == original
    assert usage().read_leases == 1
    reader.finish_read([key(1)])
    assert usage().read_leases == 0


def test_malformed_write_grant_aborts_token(
    make_manager: Factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer, peer = make_manager("writer"), make_manager("peer")
    reserve = MemoryOrchestratorClient.reserve_write

    def malformed(
        client: MemoryOrchestratorClient, entries: list[WriteRequest]
    ) -> list[WriteGrantResult]:
        return [replace(reserve(client, entries)[0], handle=None)]

    with monkeypatch.context() as patch:
        patch.setattr(MemoryOrchestratorClient, "reserve_write", malformed)
        assert (
            writer.reserve_write([key(1)], [False], LAYOUT)[key(1)][0]
            is L1Error.KEY_NOT_WRITABLE
        )
    assert peer.reserve_write([key(1)], [False], LAYOUT)[key(1)][0] is L1Error.SUCCESS


@pytest.mark.parametrize("fault", ["handle", "extent", "payload", "layout"])
def test_bad_read_grant_releases_only_new_lease(
    make_manager: Factory, usage: Usage, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    writer, reader = make_manager("writer"), make_manager("reader")
    store(writer)
    original = reader.reserve_read([key(1)])[key(1)]
    reserve = MemoryOrchestratorClient.reserve_read

    def malformed(
        client: MemoryOrchestratorClient, entries: list[ReadRequest]
    ) -> list[ReadGrantResult]:
        grant = reserve(client, entries)[0]
        handle = grant.handle
        assert handle is not None
        if fault == "handle":
            grant = replace(grant, handle=None)
        elif fault == "extent":
            grant = replace(grant, handle=replace(handle, offset=CAPACITY))
        elif fault == "payload":
            grant = replace(grant, payload_bytes=CAPACITY + 1)
        else:
            grant = replace(grant, layout=WireLayout(((4, 500),), ("float16",)))
        return [grant]

    monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", malformed)
    assert reader.reserve_read([key(1)])[key(1)] == (L1Error.KEY_NOT_EXIST, None)
    assert reader.unsafe_read([key(1)])[key(1)] == original
    assert usage().read_leases == 1
    reader.finish_read([key(1)])
    assert usage().read_leases == 0
