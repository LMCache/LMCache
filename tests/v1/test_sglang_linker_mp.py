# SPDX-License-Identifier: Apache-2.0
"""GPU IPC round trips against an independently running LMCache MP server."""

# Standard
from dataclasses import dataclass
from typing import Any
import os
import time
import uuid

# Third Party
import pytest
import torch

pytest.importorskip("sglang.srt.mem_cache.unified_cache.linker_factory")

# Third Party
from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)

# First Party
from lmcache.integration.sglang import unified_cache_linker as module

SERVER_URL = os.environ.get("LMCACHE_LINKER_TEST_SERVER")
pytestmark = [
    pytest.mark.no_shared_allocator,
    pytest.mark.skipif(
        not SERVER_URL or not torch.cuda.is_available(),
        reason="set LMCACHE_LINKER_TEST_SERVER to a CUDA MP server with chunk-size 1",
    ),
]


@dataclass
class Arguments:
    model_path: str = "test/byte-page-transport"
    revision: str = "v1"
    tp_size: int = 1


@dataclass
class Parameters:
    token_to_kv_pool_allocator: Any
    page_size: int = 2
    attn_tp_cache_group: Any = None
    tp_cache_group: Any = None
    pp_rank: int = 0
    pp_size: int = 1
    attn_cp_rank: int = 0
    attn_cp_size: int = 1


class Allocator:
    def get_kvcache(self) -> None:
        return None


def wait_for_completion(poll: Any) -> None:
    deadline = time.monotonic() + 60
    while not poll():
        assert time.monotonic() < deadline, "MP transfer did not complete"
        time.sleep(0.01)


@pytest.mark.parametrize("kind", ["mha", "mla", "swa", "mamba"])
def test_persist_across_registrations_and_restore_all_bytes(monkeypatch, kind):
    """Preserve every dtype/plane across fresh IPC registrations and GPU slots."""
    namespace = uuid.uuid4().hex
    keys = [f"page-{i}" for i in range(4)]
    names = [PoolName.KV]
    if kind == "swa":
        names.append(PoolName.SWA)
    elif kind == "mamba":
        names.append(PoolName.MAMBA)

    def build(fill):
        entries = []
        for name in names:
            specs = [(64, torch.bfloat16)]
            if kind != "mla":
                specs.append((96, torch.float32))
            buffers = [
                torch.arange(32 * width, device="cuda", dtype=torch.float32)
                .reshape(32, width)
                .to(dtype)
                .mul_(fill)
                for width, dtype in specs
            ]
            entries.append(
                DevicePoolEntry(
                    name=name,
                    indices_from_pool=name,
                    device_pool=None,
                    components=[buffers],
                    layer_mapping={0: list(range(len(buffers)))},
                    page_size=1 if name == PoolName.MAMBA else 2,
                    rows_are_pages=name == PoolName.MAMBA,
                )
            )
        group = DevicePoolGroup(entries, 1, 2)
        monkeypatch.setattr(
            module, "resolve_hybrid_device_pool_group", lambda **kw: group
        )
        return module.LMCacheLinker(
            Arguments(),
            Parameters(Allocator()),
            components=set(),
            extra_config={"server_url": SERVER_URL, "namespace": namespace},
        )

    def transfers(first_page=None):
        result = []
        for name in names:
            selected = keys
            if name == PoolName.SWA:
                selected = keys[-2:]
            elif name == PoolName.MAMBA:
                selected = keys[-1:]
            page = 1 if name == PoolName.MAMBA else 2
            result.append(
                PoolTransfer(
                    name=name,
                    keys=selected,
                    device_indices=(
                        None
                        if first_page is None
                        else torch.arange(
                            first_page * page,
                            (first_page + len(selected)) * page,
                            device="cuda",
                        )
                    ),
                    hit_policy=(
                        PoolHitPolicy.ALL_PAGES
                        if name == PoolName.KV
                        else PoolHitPolicy.TRAILING_PAGES
                    ),
                )
            )
        return result

    writer = build(1)
    expected = {
        entry.name: [t.clone() for t in entry.get_page_tensors()]
        for entry in writer.pool_group.entries
    }
    try:
        assert writer.lookup("cold", transfers()) == []
        assert writer.offload(transfers(1))
        wait_for_completion(writer.num_completed_offloads)
        assert writer.pop_completed_offload()
    finally:
        writer.close()

    reader = build(0)
    try:
        boundaries = reader.lookup("restore", transfers())
        assert boundaries == ([1, 2, 3, 4] if len(names) == 1 else [4])
        assert reader.load("restore", transfers(8))
        assert reader.start_layer_wise_loading() >= 0
        reader.layer_done_counter.wait_until(1)
        # Comparisons run on the forward stream, ordered by the imported event.
        for entry, transfer in zip(reader.pool_group.entries, transfers(), strict=True):
            count = len(transfer.keys)
            for actual, original in zip(
                entry.get_page_tensors(), expected[entry.name], strict=True
            ):
                assert torch.equal(actual[8 : 8 + count], original[1 : 1 + count])
        wait_for_completion(reader.num_completed_loads)
        assert reader.pop_completed_load() == ["restore"]
        reader.reset()
    finally:
        reader.close()
