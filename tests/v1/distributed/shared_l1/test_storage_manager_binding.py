# SPDX-License-Identifier: Apache-2.0
"""Shared/private placement, local-first lookup and backing ownership."""

# Standard
from collections.abc import Callable, Sequence
from pathlib import Path

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    EvictionConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.memory_orchestrator.api import ReadGrantResult, ReadRequest, RegionUsage
from lmcache.v1.memory_orchestrator.client import MemoryOrchestratorClient
from tests.v1.distributed.shared_l1.utils import CAPACITY, key, shared_config
from tests.v1.distributed.utils import single_row_spec
from tests.v1.memory_orchestrator.harness import Orchestrator

pytestmark = pytest.mark.no_shared_allocator


@pytest.mark.parametrize("shared_first", [False, True])
def test_shared_file_cannot_alias_private_arena(
    tmp_path: Path, orchestrator: Orchestrator, shared_first: bool
) -> None:
    region = tmp_path / "region"
    with region.open("wb") as backing:
        backing.truncate(CAPACITY)
    shared = shared_config(region, orchestrator.endpoint, "owner", size_bytes=CAPACITY)
    private = DevDaxL1ManagerConfig(
        L1MemoryManagerConfig(
            CAPACITY, use_lazy=False, shm_name="", devdax_path=str(region)
        ),
        tag="private",
    )
    configs = [shared, private] if shared_first else [private, shared]
    with pytest.raises(ValueError, match="already owned by an L1"):
        StorageManager(
            StorageManagerConfig(
                configs[0], EvictionConfig("noop"), l1_manager_configs=configs
            )
        )


@pytest.mark.parametrize("orchestrator", [4096], indirect=True)
def test_overflow_and_local_first_ownership(
    make_store: Callable[[str], StorageManager],
    usage: Callable[[], RegionUsage],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer, peer = make_store("writer"), make_store("peer")
    layout = MemoryLayoutDesc([torch.Size([1000])], [torch.uint8])
    keys = [key(1), key(2)]
    for value, k in enumerate(keys, 1):
        reserved = writer.reserve_write([k], layout)
        tensor = reserved[k].tensor
        assert tensor is not None
        tensor.fill_(value)
        writer.finish_write_by_owner(writer.prepare_write_completion(reserved))
    reserve_read = MemoryOrchestratorClient.reserve_read
    requests = []

    def record(
        client: MemoryOrchestratorClient, entries: Sequence[ReadRequest]
    ) -> list[ReadGrantResult]:
        requests.append([entry.key for entry in entries])
        return reserve_read(client, entries)

    monkeypatch.setattr(MemoryOrchestratorClient, "reserve_read", record)
    spec = single_row_spec(keys + [key(3)], layout)
    own = writer.query_prefetch_status(writer.submit_prefetch_task(spec, skip_l2=True))
    assert requests == [[key(1), key(3)]]
    remote = peer.query_prefetch_status(peer.submit_prefetch_task(spec, skip_l2=True))
    assert own is not None and remote is not None
    assert own.hit_cells[0].get_indices_list() == [0, 1]
    assert remote.hit_cells[0].get_indices_list() == [0]
    assert own.l1_owners[key(1)] != own.l1_owners[key(2)]
    with writer.read_prefetched_results(keys, own.l1_owners) as objects:
        assert objects is not None
        assert all(torch.all(obj.tensor == i) for i, obj in enumerate(objects, 1))
    writer.finish_read_prefetched(keys, l1_owners=own.l1_owners)
    peer.finish_read_prefetched(keys[:1], l1_owners=remote.l1_owners)
    calls = len(requests)
    local = writer.query_prefetch_status(
        writer.submit_prefetch_task(single_row_spec([key(2)], layout), skip_l2=True)
    )
    assert local is not None and len(requests) == calls
    writer.finish_read_prefetched([key(2)], l1_owners=local.l1_owners)
    assert usage().read_leases == 0
