# SPDX-License-Identifier: Apache-2.0
"""Unit tests for multi-holder lookup.

``RegistryTree.find_kv_all`` reports every instance holding a chunk, and
``KVController.lookup`` uses it to credit every instance still matching
the request's prefix - not only the first holder found. A router that
compares instances (for example a cache- and load-aware router) needs the
full holder set: with only the first holder reported, a prefix replicated
on two instances is indistinguishable from one held by a single instance.

The controller-level tests use the real ``RegistryTree`` so the tests
exercise the same code the production lookup path runs.
"""

# Standard
from unittest.mock import MagicMock, patch
import time

# Third Party
import pytest

# First Party
from lmcache.v1.cache_controller.controllers.kv_controller import KVController
from lmcache.v1.cache_controller.message import (
    BatchedKVOperationMsg,
    KVOpEvent,
    LookupMsg,
    OpType,
)
from lmcache.v1.cache_controller.utils import RegistryTree

LOCATION = "LocalCPUBackend"
INST_A = "instance-a"
INST_B = "instance-b"


def make_controller():
    """KVController over a real RegistryTree, with two registered
    instances of one worker each."""
    reg = RegistryTree()
    for instance_id in (INST_A, INST_B):
        reg.register_worker(
            instance_id, 0, "127.0.0.1", 8000, None, MagicMock(), time.time()
        )
    ctrl = KVController(registry=reg)
    ctrl.cluster_executor = MagicMock()
    return ctrl, reg


def admit(reg, instance_id, keys, location=LOCATION):
    """Admit `keys` for `instance_id`'s worker 0 through the real batched
    operations path."""
    msg = BatchedKVOperationMsg(
        instance_id=instance_id,
        worker_id=0,
        location=location,
        operations=[
            KVOpEvent(op_type=OpType.ADMIT, key=key, seq_num=i + 1)
            for i, key in enumerate(keys)
        ],
    )
    reg.handle_batched_kv_operations(msg)


def chunks(n, size=256, first_key=1000):
    """`n` chunk descriptors in process_tokens' (start, end, key) shape."""
    return [(i * size, (i + 1) * size, first_key + i) for i in range(n)]


async def lookup(ctrl, n_chunks):
    with patch.object(ctrl.token_database, "process_tokens") as mock_process:
        mock_process.return_value = chunks(n_chunks)
        return await ctrl.lookup(
            LookupMsg(event_id="event_123", tokens=list(range(n_chunks * 256)))
        )


class TestFindKvAll:
    """RegistryTree.find_kv_all."""

    def test_every_holder_is_reported(self):
        _, reg = make_controller()
        admit(reg, INST_A, [1000])
        admit(reg, INST_B, [1000])
        holders = reg.find_kv_all(1000)
        assert set(holders) == {INST_A, INST_B}
        assert holders[INST_A].location == LOCATION

    def test_no_holder_is_an_empty_dict(self):
        _, reg = make_controller()
        assert reg.find_kv_all(9999) == {}

    def test_exclude_instance_id_is_honoured(self):
        _, reg = make_controller()
        admit(reg, INST_A, [1000])
        admit(reg, INST_B, [1000])
        assert set(reg.find_kv_all(1000, exclude_instance_id=INST_A)) == {INST_B}

    def test_first_entry_matches_find_kv(self):
        """Wire compatibility: the dict's first entry is the instance
        find_kv returns."""
        _, reg = make_controller()
        admit(reg, INST_A, [1000])
        admit(reg, INST_B, [1000])
        first_all = next(iter(reg.find_kv_all(1000).values()))
        assert first_all == reg.find_kv(1000)


class TestMultiHolderLookup:
    """KVController.lookup over a real RegistryTree."""

    @pytest.mark.asyncio
    async def test_every_holder_is_credited_with_its_own_prefix(self):
        """A holds chunks 0-2, B holds chunks 0-1: both are reported,
        each with its own matched length."""
        ctrl, reg = make_controller()
        admit(reg, INST_A, [1000, 1001, 1002])
        admit(reg, INST_B, [1000, 1001])
        result = await lookup(ctrl, 3)
        assert result.layout_info[INST_A] == (LOCATION, 768)
        assert result.layout_info[INST_B] == (LOCATION, 512)

    @pytest.mark.asyncio
    async def test_credit_is_contiguous_from_token_zero(self):
        """B holds chunks 0 and 2 but not 1: its credit stops at the gap
        even though it holds a later chunk."""
        ctrl, reg = make_controller()
        admit(reg, INST_A, [1000, 1001, 1002])
        admit(reg, INST_B, [1000, 1002])
        result = await lookup(ctrl, 3)
        assert result.layout_info[INST_A] == (LOCATION, 768)
        assert result.layout_info[INST_B] == (LOCATION, 256)

    @pytest.mark.asyncio
    async def test_a_suffix_only_holder_is_never_credited(self):
        """B holds only chunk 1. It cannot serve any prefix of the
        request, so it earns nothing. (The previous first-match-only
        implementation could credit exactly this case: find_kv at chunk 1
        returned B once A was the only chunk-0 holder.)"""
        ctrl, reg = make_controller()
        admit(reg, INST_A, [1000, 1001])
        admit(reg, INST_B, [1001])
        result = await lookup(ctrl, 2)
        assert INST_B not in result.layout_info
        assert result.layout_info[INST_A] == (LOCATION, 512)

    @pytest.mark.asyncio
    async def test_single_holder_behaves_as_before(self):
        ctrl, reg = make_controller()
        admit(reg, INST_A, [1000, 1001, 1002])
        result = await lookup(ctrl, 3)
        assert result.layout_info == {INST_A: (LOCATION, 768)}

    @pytest.mark.asyncio
    async def test_walk_stops_when_no_instance_holds_the_next_chunk(self):
        ctrl, reg = make_controller()
        admit(reg, INST_A, [1000])
        admit(reg, INST_B, [1000])
        result = await lookup(ctrl, 3)
        assert result.layout_info[INST_A] == (LOCATION, 256)
        assert result.layout_info[INST_B] == (LOCATION, 256)

    @pytest.mark.asyncio
    async def test_miss_everywhere_is_an_empty_layout(self):
        ctrl, _ = make_controller()
        result = await lookup(ctrl, 2)
        assert result.layout_info == {}

    @pytest.mark.asyncio
    async def test_first_entry_is_the_chunk_zero_holder_find_kv_chose(self):
        """Wire compatibility for callers that read only the first entry
        (e.g. production-stack's kvaware router)."""
        ctrl, reg = make_controller()
        admit(reg, INST_A, [1000, 1001])
        admit(reg, INST_B, [1000, 1001, 1002])
        result = await lookup(ctrl, 3)
        first_instance = next(iter(result.layout_info))
        assert first_instance == reg.find_kv(1000).instance_id
